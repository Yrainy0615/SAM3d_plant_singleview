# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Geometric primitives for plant structure representation.
Defines cylinder and leaf primitives used in hybrid plant reconstruction.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class CylinderPrimitive:
    """
    Generalized cylinder primitive for stem/branch representation.
    Supports tapering (varying radius along length).
    """
    # Position and orientation
    base_position: torch.Tensor  # (3,) - starting point in 3D
    direction: torch.Tensor      # (3,) - normalized direction vector
    length: float                # scalar - cylinder length

    # Geometric parameters
    base_radius: float          # scalar - radius at base
    tip_radius: float           # scalar - radius at tip (for tapering)

    # Optional parameters
    curvature: Optional[float] = None  # scalar - optional curvature parameter

    def __post_init__(self):
        """Ensure direction is normalized."""
        if isinstance(self.direction, torch.Tensor):
            self.direction = self.direction / (torch.norm(self.direction) + 1e-8)
        else:
            self.direction = torch.tensor(self.direction, dtype=torch.float32)
            self.direction = self.direction / (torch.norm(self.direction) + 1e-8)

    def get_tip_position(self) -> torch.Tensor:
        """Calculate the tip position of the cylinder."""
        return self.base_position + self.direction * self.length

    def get_radius_at(self, t: float) -> float:
        """
        Get radius at parametric position t ∈ [0, 1].
        Linear interpolation for tapering effect.
        """
        return self.base_radius * (1 - t) + self.tip_radius * t

    def sample_surface_points(self, num_points: int = 100) -> torch.Tensor:
        """
        Sample points on cylinder surface.
        Returns (num_points, 3) tensor.
        """
        # Sample along length
        t = torch.linspace(0, 1, num_points // 10)
        # Sample around circumference
        theta = torch.linspace(0, 2 * np.pi, 10)

        points = []
        for t_i in t:
            pos = self.base_position + self.direction * (t_i * self.length)
            radius = self.get_radius_at(t_i.item())

            # Create perpendicular vectors
            perp1 = torch.tensor([self.direction[1], -self.direction[0], 0.0])
            if torch.norm(perp1) < 0.1:
                perp1 = torch.tensor([0.0, self.direction[2], -self.direction[1]])
            perp1 = perp1 / torch.norm(perp1)
            perp2 = torch.cross(self.direction, perp1)
            perp2 = perp2 / torch.norm(perp2)

            for theta_i in theta:
                offset = radius * (torch.cos(theta_i) * perp1 + torch.sin(theta_i) * perp2)
                points.append(pos + offset)

        return torch.stack(points)


@dataclass
class LeafPrimitive:
    """
    Oriented ellipsoid primitive for leaf representation.
    """
    # Position and orientation
    attachment_point: torch.Tensor  # (3,) - where leaf attaches to stem
    normal: torch.Tensor            # (3,) - leaf plane normal (points "up" from leaf)
    up_vector: torch.Tensor         # (3,) - direction along leaf length

    # Size parameters
    length: float  # scalar - leaf length
    width: float   # scalar - leaf width

    # Optional parameters
    petiole_length: Optional[float] = 0.0  # scalar - stem connecting leaf

    def __post_init__(self):
        """Ensure vectors are normalized and orthogonal."""
        if isinstance(self.normal, torch.Tensor):
            self.normal = self.normal / (torch.norm(self.normal) + 1e-8)
        else:
            self.normal = torch.tensor(self.normal, dtype=torch.float32)
            self.normal = self.normal / (torch.norm(self.normal) + 1e-8)

        if isinstance(self.up_vector, torch.Tensor):
            self.up_vector = self.up_vector / (torch.norm(self.up_vector) + 1e-8)
        else:
            self.up_vector = torch.tensor(self.up_vector, dtype=torch.float32)
            self.up_vector = self.up_vector / (torch.norm(self.up_vector) + 1e-8)

        # Ensure orthogonality
        self.up_vector = self.up_vector - torch.dot(self.up_vector, self.normal) * self.normal
        self.up_vector = self.up_vector / (torch.norm(self.up_vector) + 1e-8)

    def get_width_vector(self) -> torch.Tensor:
        """Get the width direction (orthogonal to normal and up)."""
        width_vec = torch.cross(self.normal, self.up_vector)
        return width_vec / (torch.norm(width_vec) + 1e-8)

    def get_center_position(self) -> torch.Tensor:
        """Get the geometric center of the leaf."""
        return self.attachment_point + self.up_vector * (self.length / 2)

    def sample_surface_points(self, num_points: int = 50) -> torch.Tensor:
        """
        Sample points on leaf surface (modeled as ellipse).
        Returns (num_points, 3) tensor.
        """
        # Parametric ellipse in 2D
        theta = torch.linspace(0, 2 * np.pi, num_points)

        width_vec = self.get_width_vector()
        center = self.get_center_position()

        points = []
        for t in theta:
            # Ellipse equation
            u = (self.length / 2) * torch.cos(t)
            v = (self.width / 2) * torch.sin(t)

            point = center + u * self.up_vector + v * width_vec
            points.append(point)

        return torch.stack(points)

    def to_oriented_bbox(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get oriented bounding box representation.
        Returns (center, extents) where extents = [length, width, thickness].
        """
        center = self.get_center_position()
        extents = torch.tensor([self.length, self.width, 0.01])  # Thin in normal direction
        return center, extents


def create_cylinder_from_endpoints(
    start: torch.Tensor,
    end: torch.Tensor,
    start_radius: float,
    end_radius: float
) -> CylinderPrimitive:
    """
    Convenient factory function to create cylinder from two endpoints.
    """
    direction = end - start
    length = torch.norm(direction).item()
    direction = direction / length

    return CylinderPrimitive(
        base_position=start,
        direction=direction,
        length=length,
        base_radius=start_radius,
        tip_radius=end_radius
    )


def create_leaf_at_node(
    node_position: torch.Tensor,
    stem_direction: torch.Tensor,
    phyllotactic_angle: float,
    leaf_length: float,
    leaf_width: float,
    leaf_angle: float = 45.0  # degrees from horizontal
) -> LeafPrimitive:
    """
    Create leaf primitive at a stem node with proper phyllotaxis.

    Args:
        node_position: 3D position where leaf attaches
        stem_direction: Direction of stem at attachment point
        phyllotactic_angle: Angle around stem (degrees, 0-360)
        leaf_length: Length of leaf
        leaf_width: Width of leaf
        leaf_angle: Angle of leaf from horizontal (degrees)
    """
    # Convert to radians
    phyllo_rad = np.deg2rad(phyllotactic_angle)
    leaf_rad = np.deg2rad(leaf_angle)

    # Create perpendicular basis
    stem_dir = stem_direction / (torch.norm(stem_direction) + 1e-8)

    # Find perpendicular vector
    perp = torch.tensor([stem_dir[1], -stem_dir[0], 0.0])
    if torch.norm(perp) < 0.1:
        perp = torch.tensor([0.0, stem_dir[2], -stem_dir[1]])
    perp = perp / torch.norm(perp)

    # Second perpendicular
    perp2 = torch.cross(stem_dir, perp)
    perp2 = perp2 / torch.norm(perp2)

    # Rotate around stem by phyllotactic angle
    radial_dir = torch.cos(torch.tensor(phyllo_rad)) * perp + torch.sin(torch.tensor(phyllo_rad)) * perp2

    # Leaf up-vector (tilted from radial direction)
    up_vector = torch.cos(torch.tensor(leaf_rad)) * radial_dir + torch.sin(torch.tensor(leaf_rad)) * stem_dir

    # Normal points "out" from leaf surface
    normal = torch.cross(stem_dir, up_vector)
    normal = normal / (torch.norm(normal) + 1e-8)

    return LeafPrimitive(
        attachment_point=node_position,
        normal=normal,
        up_vector=up_vector,
        length=leaf_length,
        width=leaf_width
    )
