from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class TemporalStyleConfig:
    style_dim: int = 192
    max_gate: float = 0.25
    max_norm_ratio: float = 0.10
    strength: float = 1.0
    min_confidence: float = 0.45
    min_source_f0_confidence: float = 0.35
    min_patch_f0_confidence: float = 0.50
    min_patch_quality: float = 0.90
    min_weight_margin: float = 0.015
    max_register_distance: float = 0.20
    max_voiced_ratio_distance: float = 0.35
    require_f0_valid_match: bool = True
    smoothing_seconds: float = 0.50

    def validate(self) -> None:
        if self.style_dim <= 0:
            raise ValueError("style_dim must be positive")
        if not 0.0 <= self.max_gate <= 1.0:
            raise ValueError("max_gate must be between 0 and 1")
        if not 0.0 <= self.max_norm_ratio <= 1.0:
            raise ValueError("max_norm_ratio must be between 0 and 1")
        if self.strength < 0:
            raise ValueError("strength must be non-negative")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        bounded = (
            ("min_source_f0_confidence", self.min_source_f0_confidence),
            ("min_patch_f0_confidence", self.min_patch_f0_confidence),
            ("min_patch_quality", self.min_patch_quality),
            ("min_weight_margin", self.min_weight_margin),
            ("max_register_distance", self.max_register_distance),
            ("max_voiced_ratio_distance", self.max_voiced_ratio_distance),
        )
        for name, value in bounded:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.smoothing_seconds < 0:
            raise ValueError("smoothing_seconds must be non-negative")


class TemporalStyleMerge(nn.Module):
    """Replace only the repeated style slice before Seed-VC's merge layer."""

    def __init__(self, base: nn.Module, style_dim: int = 192):
        super().__init__()
        self.base = base
        self.style_dim = style_dim
        self._schedule: torch.Tensor | None = None

    def set_schedule(self, schedule: torch.Tensor | None) -> None:
        if schedule is not None:
            if schedule.ndim != 3 or schedule.size(0) != 1 or schedule.size(-1) != self.style_dim:
                raise ValueError(f"temporal style schedule must be [1, T, {self.style_dim}]")
            schedule = schedule.detach()
        self._schedule = schedule

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        if self._schedule is None or x_in.size(-1) < self.style_dim:
            return self.base(x_in)
        schedule = self._schedule.to(device=x_in.device, dtype=x_in.dtype)
        if schedule.size(1) != x_in.size(1):
            schedule = F.interpolate(
                schedule.transpose(1, 2),
                size=x_in.size(1),
                mode="linear",
                align_corners=True,
            ).transpose(1, 2)
        schedule = schedule.expand(x_in.size(0), -1, -1)
        original_style = x_in[..., -self.style_dim :]
        # CFG stacks a zero-style null branch. Preserve it instead of leaking
        # target style into the unconditional estimate.
        conditioned = original_style.abs().sum(dim=(1, 2), keepdim=True) > 1e-8
        replacement = torch.where(conditioned, schedule, original_style)
        merged_input = torch.cat([x_in[..., : -self.style_dim], replacement], dim=-1)
        return self.base(merged_input)


def install_temporal_style_merge(seed_model: Any, *, style_dim: int = 192) -> TemporalStyleMerge:
    estimator = seed_model.cfm.estimator
    current = estimator.cond_x_merge_linear
    if isinstance(current, TemporalStyleMerge):
        return current
    wrapper = TemporalStyleMerge(current, style_dim=style_dim)
    estimator.cond_x_merge_linear = wrapper
    return wrapper


def load_temporal_memory_records(memory: str | Path) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    root = Path(memory).resolve()
    if root.is_file():
        root = root.parent
    metadata_path = root / "memory.json"
    records_path = root / "memory.jsonl"
    if not metadata_path.is_file() or not records_path.is_file():
        raise FileNotFoundError(f"Temporal memory is incomplete: {root}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("memory_type") != "mosaic_temporal_timbre_memory":
        raise ValueError(f"Unsupported temporal memory type: {metadata.get('memory_type')}")
    records: dict[str, dict[str, Any]] = {}
    with records_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid temporal memory JSON at line {line_number}: {exc}") from exc
            records[str(record["patch_id"])] = record
    return root, metadata, records


def load_temporal_query(query: str | Path) -> list[dict[str, Any]]:
    path = Path(query).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Temporal query does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid temporal query JSON at line {line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"Temporal query has no frames: {path}")
    return rows


def _moving_average(values: torch.Tensor, window: int) -> torch.Tensor:
    if window <= 1 or values.size(0) <= 1:
        return values
    window = min(window, values.size(0))
    if window % 2 == 0 and window > 1:
        window -= 1
    if window <= 1:
        return values
    radius = window // 2
    padded = F.pad(values.transpose(0, 1).unsqueeze(0), (radius, radius), mode="replicate")
    return F.avg_pool1d(padded, kernel_size=window, stride=1).squeeze(0).transpose(0, 1)


def _query_interval(rows: list[dict[str, Any]]) -> float:
    times = [float(row.get("source_time_seconds", 0.0)) for row in rows]
    differences = [right - left for left, right in zip(times, times[1:]) if right > left]
    if not differences:
        return 0.10
    differences.sort()
    return differences[len(differences) // 2]


def _row_gate_reasons(
    row: dict[str, Any],
    records: dict[str, dict[str, Any]],
    config: TemporalStyleConfig,
) -> list[str]:
    reasons = [str(reason) for reason in row.get("gate_reasons", [])]
    if row.get("gate_accepted") is False and not reasons:
        reasons.append("upstream_gate")

    confidence = max(0.0, min(1.0, float(row.get("retrieval_confidence", 0.0))))
    if confidence < config.min_confidence:
        reasons.append("retrieval_confidence")

    selected = str(row.get("selected_patch_id") or "")
    if not selected or selected not in records:
        reasons.append("no_candidate")
        return list(dict.fromkeys(reasons))

    candidates = row.get("candidates", [])
    candidate = next(
        (item for item in candidates if str(item.get("patch_id", "")) == selected),
        None,
    )
    weights = sorted(
        (max(0.0, float(item.get("soft_weight", 0.0))) for item in candidates),
        reverse=True,
    )
    weight_margin = weights[0] - weights[1] if len(weights) > 1 else (weights[0] if weights else 0.0)
    if weight_margin < config.min_weight_margin:
        reasons.append("candidate_margin")

    source = row.get("source_features", {})
    target = candidate.get("target_features", {}) if candidate else records[selected].get("key", {})
    quality = (
        candidate.get("quality_score")
        if candidate and candidate.get("quality_score") is not None
        else records[selected].get("quality", {}).get("quality_score")
    )
    if quality is not None and float(quality) < config.min_patch_quality:
        reasons.append("patch_quality")

    source_valid = source.get("f0_valid")
    target_valid = target.get("f0_valid")
    if (
        config.require_f0_valid_match
        and source_valid is not None
        and target_valid is not None
        and bool(source_valid) != bool(target_valid)
    ):
        reasons.append("f0_valid_mismatch")
    if bool(source_valid) and source.get("f0_confidence") is not None:
        if float(source["f0_confidence"]) < config.min_source_f0_confidence:
            reasons.append("source_f0_confidence")
    if bool(target_valid) and target.get("f0_confidence") is not None:
        if float(target["f0_confidence"]) < config.min_patch_f0_confidence:
            reasons.append("patch_f0_confidence")

    if source.get("relative_register") is not None and target.get("relative_register") is not None:
        if abs(float(source["relative_register"]) - float(target["relative_register"])) > config.max_register_distance:
            reasons.append("register_distance")
    if source.get("voiced_ratio") is not None and target.get("voiced_ratio") is not None:
        if abs(float(source["voiced_ratio"]) - float(target["voiced_ratio"])) > config.max_voiced_ratio_distance:
            reasons.append("voiced_ratio_distance")
    return list(dict.fromkeys(reasons))


def _candidate_embedding(
    row: dict[str, Any],
    records: dict[str, dict[str, Any]],
    embedding_loader: Callable[[dict[str, Any]], torch.Tensor],
    cache: dict[str, torch.Tensor],
    canonical: torch.Tensor,
) -> torch.Tensor | None:
    weighted: list[tuple[float, torch.Tensor]] = []
    for candidate in row.get("candidates", []):
        patch_id = str(candidate.get("patch_id", ""))
        if patch_id not in records:
            continue
        weight = max(0.0, float(candidate.get("soft_weight", 0.0)))
        if weight <= 0:
            continue
        if patch_id not in cache:
            embedding = embedding_loader(records[patch_id]).detach().float().cpu().reshape(-1)
            if embedding.numel() != canonical.numel():
                raise ValueError(
                    f"Temporal patch {patch_id} style has {embedding.numel()} dimensions; "
                    f"expected {canonical.numel()}"
                )
            cache[patch_id] = embedding
        weighted.append((weight, cache[patch_id]))
    if not weighted:
        selected = str(row.get("selected_patch_id") or "")
        if selected not in records:
            return None
        if selected not in cache:
            cache[selected] = embedding_loader(records[selected]).detach().float().cpu().reshape(-1)
        return cache[selected]
    total = sum(weight for weight, _ in weighted)
    return sum((weight / total) * embedding for weight, embedding in weighted)


def build_temporal_style_schedule(
    query: str | Path,
    memory: str | Path,
    canonical_style: torch.Tensor,
    embedding_loader: Callable[[dict[str, Any]], torch.Tensor],
    *,
    source_frames: int,
    config: TemporalStyleConfig | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build a bounded per-frame style schedule from TTM retrieval output."""
    config = config or TemporalStyleConfig()
    config.validate()
    if source_frames <= 0:
        raise ValueError("source_frames must be positive")
    _, _, records = load_temporal_memory_records(memory)
    rows = load_temporal_query(query)
    canonical = canonical_style.detach().float().cpu().reshape(-1)
    if canonical.numel() != config.style_dim:
        raise ValueError(f"canonical style must have {config.style_dim} dimensions")

    cache: dict[str, torch.Tensor] = {}
    query_styles: list[torch.Tensor] = []
    gates: list[float] = []
    change_ratios: list[float] = []
    gate_rejections: dict[str, int] = {}
    base_norm = canonical.norm().clamp_min(1e-6)
    max_delta_norm = base_norm * config.max_norm_ratio
    for row in rows:
        confidence = max(0.0, min(1.0, float(row.get("retrieval_confidence", 0.0))))
        selected = row.get("selected_patch_id")
        rejection_reasons = _row_gate_reasons(row, records, config)
        for reason in rejection_reasons:
            gate_rejections[reason] = gate_rejections.get(reason, 0) + 1
        target = None
        if selected and not rejection_reasons:
            target = _candidate_embedding(row, records, embedding_loader, cache, canonical)
        if target is None:
            query_styles.append(canonical.clone())
            gates.append(0.0)
            change_ratios.append(0.0)
            continue
        delta = target - canonical
        delta_norm = delta.norm().clamp_min(1e-6)
        delta = delta * torch.minimum(torch.ones_like(delta_norm), max_delta_norm / delta_norm)
        gate = min(config.max_gate, config.max_gate * config.strength * confidence)
        style = canonical + gate * delta
        query_styles.append(style)
        gates.append(gate)
        change_ratios.append(float((style - canonical).norm() / base_norm))

    styles = torch.stack(query_styles, dim=0)
    interval = _query_interval(rows)
    smoothing_window = max(1, round(config.smoothing_seconds / max(interval, 1e-6)))
    styles = _moving_average(styles, smoothing_window)
    schedule = F.interpolate(
        styles.transpose(0, 1).unsqueeze(0),
        size=source_frames,
        mode="linear",
        align_corners=True,
    ).transpose(1, 2)
    finite = bool(torch.isfinite(schedule).all())
    if not finite:
        raise ValueError("Temporal style schedule contains non-finite values")
    summary = {
        "schema_version": 2,
        "mode": "temporal_style_schedule_p1",
        "query_frames": len(rows),
        "source_frames": source_frames,
        "active_query_frames": sum(gate > 0 for gate in gates),
        "unique_patches_embedded": len(cache),
        "mean_gate": sum(gates) / len(gates),
        "max_gate": max(gates),
        "mean_style_change_ratio": sum(change_ratios) / len(change_ratios),
        "max_style_change_ratio": max(change_ratios),
        "query_interval_seconds": interval,
        "smoothing_window_frames": smoothing_window,
        "gate_rejections": gate_rejections,
        "config": asdict(config),
    }
    return schedule, _json_safe(summary)


def save_temporal_style_summary(summary: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
