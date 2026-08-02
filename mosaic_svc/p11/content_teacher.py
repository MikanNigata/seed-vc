from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass
class ContentTeacherConfig:
    input_dim: int = 768
    hidden_dim: int = 256
    bottleneck_dim: int = 128
    layers: int = 2
    heads: int = 4
    kernel_size: int = 5
    max_whisper_gate: float = 0.30
    initial_whisper_gate: float = 0.10
    max_adapter_scale: float = 0.25
    initial_adapter_scale: float = 0.05
    dropout: float = 0.05


def _bounded_parameter(initial: float, maximum: float) -> nn.Parameter:
    ratio = min(max(initial / max(maximum, 1e-8), 1e-6), 1.0 - 1e-6)
    return nn.Parameter(torch.logit(torch.tensor(ratio)))


class GatedContentFusion(nn.Module):
    """ContentVec main path with a bounded Whisper semantic anchor."""

    def __init__(self, config: ContentTeacherConfig):
        super().__init__()
        self.config = config
        self.contentvec_projection = nn.Linear(config.input_dim, config.input_dim, bias=False)
        self.whisper_projection = nn.Linear(config.input_dim, config.input_dim, bias=False)
        self.gate_logit = _bounded_parameter(config.initial_whisper_gate, config.max_whisper_gate)
        nn.init.eye_(self.contentvec_projection.weight)
        nn.init.eye_(self.whisper_projection.weight)

    @property
    def whisper_gate(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit) * self.config.max_whisper_gate

    def forward(self, contentvec: torch.Tensor, whisper: torch.Tensor) -> torch.Tensor:
        if whisper.size(1) != contentvec.size(1):
            whisper = nn.functional.interpolate(
                whisper.transpose(1, 2), size=contentvec.size(1), mode="linear", align_corners=False
            ).transpose(1, 2)
        return self.contentvec_projection(contentvec) + self.whisper_gate * self.whisper_projection(whisper)


class DeTimbreBlock(nn.Module):
    def __init__(self, config: ContentTeacherConfig):
        super().__init__()
        h = config.hidden_dim
        self.attention_norm = nn.LayerNorm(h)
        self.attention = nn.MultiheadAttention(h, config.heads, dropout=config.dropout, batch_first=True)
        self.conv_norm = nn.LayerNorm(h)
        self.depthwise = nn.Conv1d(h, h, config.kernel_size, padding=config.kernel_size // 2, groups=h)
        self.pointwise = nn.Conv1d(h, h, 1)
        self.ffn_norm = nn.LayerNorm(h)
        self.ffn = nn.Sequential(
            nn.Linear(h, config.bottleneck_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.bottleneck_dim, h),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        normalized = self.attention_norm(x)
        attended, _ = self.attention(normalized, normalized, normalized, key_padding_mask=padding_mask, need_weights=False)
        x = x + self.dropout(attended)
        convolved = self.conv_norm(x).transpose(1, 2)
        convolved = self.pointwise(nn.functional.silu(self.depthwise(convolved))).transpose(1, 2)
        x = x + self.dropout(convolved)
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


class DeTimbreAdapter(nn.Module):
    """Bounded residual adapter trained to remove timbre variation while retaining content."""

    def __init__(self, config: ContentTeacherConfig):
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
        self.blocks = nn.ModuleList([DeTimbreBlock(config) for _ in range(config.layers)])
        self.output_projection = nn.Linear(config.hidden_dim, config.input_dim)
        self.scale_logit = _bounded_parameter(config.initial_adapter_scale, config.max_adapter_scale)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    @property
    def scale(self) -> torch.Tensor:
        return torch.sigmoid(self.scale_logit) * self.config.max_adapter_scale

    def forward(self, content: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.input_projection(content)
        for block in self.blocks:
            hidden = block(hidden, padding_mask)
        return content + self.scale * self.output_projection(hidden)


class ContentTeacher(nn.Module):
    def __init__(self, config: ContentTeacherConfig | None = None):
        super().__init__()
        self.config = config or ContentTeacherConfig()
        self.fusion = GatedContentFusion(self.config)
        self.detimbre = DeTimbreAdapter(self.config)

    def forward(
        self,
        contentvec: torch.Tensor,
        whisper: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.detimbre(self.fusion(contentvec, whisper), padding_mask)


def save_content_teacher(model: ContentTeacher, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.config), "state_dict": model.state_dict()}, path)


def load_content_teacher(path: str | Path, device: torch.device | str = "cpu") -> ContentTeacher:
    checkpoint = torch.load(path, map_location="cpu")
    model = ContentTeacher(ContentTeacherConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device)
