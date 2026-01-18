# Quick Start: SAM 3 + SAM3D Integration

## ✅ TODO
- ✅ SAM 3 repository cloned to `/sam3/`
- ✅ build SAM3 with `cd sam3 && pip install -e .`
- ✅ SAM3D checkpoints downloaded to `/checkpoints/hf/`

## 🚀 Usage (3 Steps)

### 1. Request SAM 3 Access (Required - One Time)

SAM 3 is a gated model. You need approval:

1. Visit: https://huggingface.co/facebook/sam3
2. Click "Request Access"
3. Wait for approval email (usually 1-2 hours to 1 day)
4. Run: `huggingface-cli login` with your HuggingFace token

### 2. Run the Integration

Once you have SAM 3 access:

```bash
python demo_sam3.py \
    --image_path "path/to/image.jpg" \
    --prompt "potted plant" \
    --output_dir "outputs/plants"
```

### 3. View Results

Results are saved to your output directory:
- `object_000_splat.ply` - 3D Gaussian splat
- `object_000_mesh.obj` - 3D mesh
- `object_000_mask.png` - Segmentation mask
- `object_000_rgba.png` - RGBA image

## 📝 Example Commands


### Reconstruct Toys
```bash
python demo_sam3.py \
    --image_path "notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png" \
    --prompt "toy" \
    --output_dir "outputs/toys"
```


## 🔧 All Options

| Option | Description | Default | Required |
|--------|-------------|---------|----------|
| `--image_path` | Path to input image | - | ✅ Yes |
| `--prompt` | Text description (e.g., "plant", "toy") | - | ✅ Yes |
| `--output_dir` | Output directory | `outputs/sam3d_sam3` | No |
| `--config_path` | SAM3D config path | `checkpoints/hf/pipeline.yaml` | No |
| `--compile` | Compile SAM3D for speed | False | No |
| `--seed` | Random seed | 42 | No |

