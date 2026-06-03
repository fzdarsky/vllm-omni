# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Patch profile configuration for V-JEPA tubelet extraction.

Defines the spatial and temporal patch sizes for different model architectures.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PatchProfile:
    """Patch format configuration for patchifying video frames into tubelets.

    Attributes:
        name: Profile identifier
        temporal_size: Number of frames per tubelet (t dimension)
        spatial_size: Spatial patch size (h, w)
    """
    name: str
    temporal_size: int
    spatial_size: tuple[int, int]

    @classmethod
    def vjepa(cls) -> "PatchProfile":
        """V-JEPA default profile: 2 frames, 16x16 spatial."""
        return cls(name="vjepa", temporal_size=2, spatial_size=(16, 16))

    @classmethod
    def vjepa_large(cls) -> "PatchProfile":
        """V-JEPA large model profile."""
        return cls(name="vjepa_large", temporal_size=2, spatial_size=(14, 14))

    @classmethod
    def clip(cls) -> "PatchProfile":
        """CLIP-style profile: single frame, 14x14 spatial."""
        return cls(name="clip", temporal_size=1, spatial_size=(14, 14))

    @classmethod
    def from_name(cls, name: str) -> "PatchProfile":
        """Get profile by name."""
        profiles = {
            "vjepa": cls.vjepa,
            "vjepa_large": cls.vjepa_large,
            "clip": cls.clip,
        }
        if name not in profiles:
            raise ValueError(f"Unknown profile: {name}. Available: {list(profiles.keys())}")
        return profiles[name]()
