from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass
class PrototypeMeta:
    name: str
    path: str
    quality: float = 1.0
    category: str = "neutral_mid"
    approved: bool = True


class PrototypeBank:
    """Stores safe L1 prototypes as CAMPPlus style vectors only."""

    def __init__(self, embeddings: torch.Tensor, metas: list[PrototypeMeta], canonical: torch.Tensor | None = None):
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be [N, D]")
        if embeddings.size(0) != len(metas):
            raise ValueError("metadata count must match embedding count")
        self.embeddings = embeddings.float().cpu()
        self.metas = metas
        self.canonical = canonical.float().cpu() if canonical is not None else None

    @property
    def dim(self) -> int:
        return int(self.embeddings.size(1))

    def robust_centroid(self, keep_ratio: float = 0.80) -> torch.Tensor:
        emb = self.embeddings
        if emb.size(0) == 1:
            return emb[0]
        center = emb.mean(dim=0, keepdim=True)
        dist = torch.cdist(F.normalize(emb, dim=-1), F.normalize(center, dim=-1)).squeeze(1)
        keep = max(1, int(round(emb.size(0) * keep_ratio)))
        idx = torch.argsort(dist)[:keep]
        weights = torch.tensor([self.metas[i].quality for i in idx.tolist()], dtype=emb.dtype).clamp_min(0.01)
        return (emb[idx] * weights[:, None]).sum(dim=0) / weights.sum()

    def corrected_style(
        self,
        canonical: torch.Tensor | None = None,
        max_norm_ratio: float = 0.10,
        max_gate: float = 0.25,
        strength: float = 1.0,
    ) -> torch.Tensor:
        base = canonical.float().cpu() if canonical is not None else self.canonical
        if base is None:
            raise ValueError("canonical style is required")
        if base.ndim == 2:
            base_vec = base[0]
        else:
            base_vec = base
        centroid = self.robust_centroid()
        delta = centroid - base_vec
        max_norm = base_vec.norm().clamp_min(1e-6) * max_norm_ratio
        delta_norm = delta.norm().clamp_min(1e-6)
        delta = delta * torch.minimum(torch.ones_like(delta_norm), max_norm / delta_norm)
        gate = max(0.0, min(float(max_gate), float(max_gate) * float(strength)))
        out = base_vec + gate * delta
        return out.unsqueeze(0)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "embeddings": self.embeddings,
                "canonical": self.canonical,
                "metas": [asdict(meta) for meta in self.metas],
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PrototypeBank":
        payload = torch.load(str(path), map_location="cpu")
        metas = [PrototypeMeta(**item) for item in payload["metas"]]
        return cls(payload["embeddings"], metas, payload.get("canonical"))


def build_bank(items: Iterable[tuple[torch.Tensor, PrototypeMeta]], canonical: torch.Tensor | None = None) -> PrototypeBank:
    embeddings = []
    metas = []
    for emb, meta in items:
        if not meta.approved:
            continue
        embeddings.append(emb.squeeze(0).float().cpu())
        metas.append(meta)
    if not embeddings:
        raise ValueError("no approved prototypes")
    return PrototypeBank(torch.stack(embeddings, dim=0), metas, canonical)
