from __future__ import annotations

from pathlib import Path

import torch

from mosaic_svc.p0.prototype_bank import PrototypeBank


def load_conditioned_style(
    identity_profile: str | Path,
    device: torch.device | str,
    prototype_bank: str | Path | None = None,
    prototype_strength: float = 1.0,
    prototype_max_norm_ratio: float = 0.10,
    prototype_max_gate: float = 0.25,
) -> torch.Tensor:
    """Load the canonical identity and apply the same bounded L1 correction everywhere."""
    profile = torch.load(identity_profile, map_location="cpu")
    if "centroid" not in profile:
        raise ValueError(f"identity profile has no centroid: {identity_profile}")
    style = profile["centroid"].float().view(1, -1)
    if prototype_bank:
        bank = PrototypeBank.load(prototype_bank)
        if bank.dim != style.size(-1):
            raise ValueError(
                f"prototype style dimension {bank.dim} does not match identity dimension {style.size(-1)}"
            )
        style = bank.corrected_style(
            canonical=style,
            max_norm_ratio=prototype_max_norm_ratio,
            max_gate=prototype_max_gate,
            strength=prototype_strength,
        )
    return style.to(device)
