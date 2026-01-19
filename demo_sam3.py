# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
SAM 3 + SAM3D Integration Demo
Given a single image, uses SAM 3 to generate masks (via text prompt or auto generation),
then feeds each mask to SAM3D for 3D reconstruction, saving each result as a separate model.
"""

import sys
import os
import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path

# Import SAM 3
sys.path.append("sam3")

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Import SAM3D inference code
sys.path.append("notebook")
from inference import Inference, load_image


def setup_sam3(device="cuda"):
    """Initialize SAM 3 model and processor."""
    print("Loading SAM 3 model...")
    model = build_sam3_image_model()
    model = model.to(device)
    processor = Sam3Processor(model)
    print("SAM 3 loaded successfully!")
    return model, processor


def setup_sam3d(config_path="checkpoints/hf/pipeline.yaml", compile=False):
    """Initialize SAM3D inference pipeline."""
    print("Loading SAM3D model...")
    inference = Inference(config_path, compile=compile)
    print("SAM3D loaded successfully!")
    return inference


def get_masks_from_sam3(processor, image_path, prompt):
    """
    Get segmentation masks from SAM 3.

    Args:
        processor: SAM3Processor instance
        image_path: Path to input image
        prompt: Text prompt for segmentation (string or list of strings)

    Returns:
        masks: Tensor of binary masks (N, H, W)
        boxes: Tensor of bounding boxes (N, 4) in [x1, y1, x2, y2] format
        scores: Tensor of confidence scores (N,)
        image: PIL Image
    """
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Handle both single prompt (string) and multiple prompts (list)
    prompts = [prompt] if isinstance(prompt, str) else prompt

    all_masks = []
    all_boxes = []
    all_scores = []

    for p in prompts:
        # Set image in processor
        print(f"Processing image: {image_path}")
        inference_state = processor.set_image(image)

        # Run SAM 3 with text prompt
        print(f"Prompting SAM 3 with text: '{p}'")
        output = processor.set_text_prompt(state=inference_state, prompt=p)

        # Extract results
        masks = output["masks"].squeeze(1)  # Shape: (N, H, W) - N masks
        boxes = output["boxes"]  # Shape: (N, 4) - N bounding boxes
        scores = output["scores"]  # Shape: (N,) - N confidence scores

        print(f"SAM 3 detected {len(masks)} object(s) for prompt '{p}'")

        all_masks.append(masks)
        all_boxes.append(boxes)
        all_scores.append(scores)

    # Concatenate all results
    all_masks = torch.cat(all_masks, dim=0)
    all_boxes = torch.cat(all_boxes, dim=0)
    all_scores = torch.cat(all_scores, dim=0)

    print(f"Total SAM 3 detected {len(all_masks)} object(s) across all prompts")

    return all_masks, all_boxes, all_scores, image


def create_rgba_from_mask(image, mask):
    """
    Create RGBA image with mask in alpha channel.

    Args:
        image: PIL Image (RGB)
        mask: Binary mask (H, W) - can be torch.Tensor or numpy array

    Returns:
        PIL Image (RGBA)
    """
    # Convert to numpy
    img_np = np.array(image)  # (H, W, 3)

    # Convert mask to numpy if it's a tensor
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask

    # Ensure mask is binary (0 or 255)
    mask_binary = (mask_np > 0.5).astype(np.uint8) * 255  # (H, W)

    # Create RGBA
    rgba = np.zeros((img_np.shape[0], img_np.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = img_np
    rgba[:, :, 3] = mask_binary

    # Return RGBA image (mode parameter is deprecated, but PIL auto-detects)
    return Image.fromarray(rgba)


def run_sam3d_on_mask(sam3d_inference, image, mask, seed=42, enable_mesh=True):
    """
    Run SAM3D reconstruction on a single masked object.

    Args:
        sam3d_inference: SAM3D Inference instance
        image: PIL Image (RGB)
        mask: Binary mask (H, W) - can be torch.Tensor or numpy array
        seed: Random seed
        enable_mesh: Whether to generate mesh output

    Returns:
        output: SAM3D output dict containing reconstruction results
    """
    # Convert image to numpy array (SAM3D's merge_mask_to_rgba expects numpy array)
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image

    # Convert mask to numpy if it's a tensor
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask

    # Prepare RGBA for SAM3D
    rgba_image = sam3d_inference.merge_mask_to_rgba(image_np, mask_np)

    # Run SAM3D pipeline with mesh generation enabled
    output = sam3d_inference._pipeline.run(
        rgba_image,
        None,
        seed,
        stage1_only=False,
        with_mesh_postprocess=enable_mesh,
        with_texture_baking=False,
        with_layout_postprocess=True,
        use_vertex_color=True,
        stage1_inference_steps=None,
        pointmap=None,
    )

    return output


def main(args):
    """Main integration pipeline."""

    # Setup models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam3_model, sam3_processor = setup_sam3(device=device)
    sam3d_inference = setup_sam3d(config_path=args.config_path, compile=args.compile)

    # Get masks from SAM 3
    masks, boxes, scores, image = get_masks_from_sam3(
        sam3_processor,
        args.image_path,
        prompt=args.prompt
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each mask with SAM3D
    for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        print(f"\n--- Processing object {i+1}/{len(masks)} (score: {score:.3f}) ---")

        # Run SAM3D
        try:
            output = run_sam3d_on_mask(
                sam3d_inference, image, mask, 
                seed=args.seed + i, 
                enable_mesh=args.enable_mesh
            )

            # Save Gaussian splat
            gs_path = output_dir / f"object_{i:03d}_splat.ply"
            output["gs"].save_ply(str(gs_path))
            print(f"Saved Gaussian splat to: {gs_path}")
            mesh_path = output_dir / f"object_{i:03d}_mesh.ply"
            output["glb"].export(str(mesh_path))
            print(f"Saved  mesh to: {mesh_path}")
           
            # Save mask visualization
            mask_path = output_dir / f"object_{i:03d}_mask.png"
            # Convert tensor to numpy if needed
            mask_np = mask.cpu().numpy()
            mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
            mask_img.save(mask_path)
            print(f"Saved mask to: {mask_path}")

            # Save RGBA crop (full image with alpha channel)
            rgba_path = output_dir / f"object_{i:03d}_rgba.png"
            rgba = create_rgba_from_mask(image, mask)
            rgba.save(rgba_path)
            print(f"Saved RGBA image to: {rgba_path}")

            # Save cropped RGB (bounding box crop)
            crop_path = output_dir / f"object_{i:03d}_crop.png"
            # Convert box tensor to integers
            x1, y1, x2, y2 = [int(coord) for coord in box.cpu().numpy()]
            # Crop the image to bounding box
            cropped_image = image.crop((x1, y1, x2, y2))
            cropped_image.save(crop_path)
            print(f"Saved cropped RGB to: {crop_path}")

        except Exception as e:
            print(f"Error processing object {i+1}: {e}")
            continue

    print(f"\nAll done! Results saved to: {output_dir}")
    print(f"Total objects reconstructed: {len(masks)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM 3 + SAM3D Integration Demo")

    # Input/Output
    parser.add_argument("--image_path", type=str, default='/mnt/data/gaussianplant_data/newplant3/images/IMG_6796.JPG',
                        help="Path to input image")
    parser.add_argument("--output_dir", type=str, default="outputs/newplant3/",
                        help="Directory to save outputs")

    # SAM 3 configuration
    parser.add_argument("--prompt", type=str, nargs='+', default=['leaf','branch'],
                        help="Text prompt(s) for SAM 3. Can be single string or multiple strings (e.g., 'potted plant' or 'leaf' 'branch')")

    # SAM3D configuration
    parser.add_argument("--config_path", type=str, default="checkpoints/hf/pipeline.yaml",
                        help="Path to SAM3D config")
    parser.add_argument("--compile", action="store_true",
                        help="Compile SAM3D model for faster inference")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for SAM3D")
    parser.add_argument("--enable_mesh", action="store_true",
                        help="Enable mesh generation (requires more computation)")

    args = parser.parse_args()

    main(args)
