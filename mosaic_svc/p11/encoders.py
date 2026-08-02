from __future__ import annotations

import torch
from torch import nn
from transformers import AutoFeatureExtractor, AutoModel, WhisperModel


class FrozenTeacherEncoders(nn.Module):
    def __init__(
        self,
        contentvec_path: str,
        whisper_name: str = "openai/whisper-small",
        device: torch.device | str = "cuda",
        fp16: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device)
        dtype = torch.float16 if fp16 and self.device.type == "cuda" else torch.float32
        self.contentvec = AutoModel.from_pretrained(contentvec_path).to(self.device).eval()
        self.whisper = WhisperModel.from_pretrained(whisper_name, torch_dtype=dtype).to(self.device).eval()
        del self.whisper.decoder
        self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)
        for module in (self.contentvec, self.whisper):
            for parameter in module.parameters():
                parameter.requires_grad = False

    @torch.inference_mode()
    def forward(self, waves_16k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        waves_16k = waves_16k.to(self.device)
        contentvec = self.contentvec(waves_16k).last_hidden_state.float()
        inputs = self.whisper_feature_extractor(
            [wave.detach().cpu().numpy() for wave in waves_16k],
            return_tensors="pt",
            return_attention_mask=True,
            sampling_rate=16000,
        )
        features = inputs.input_features.to(self.device, dtype=self.whisper.encoder.dtype)
        attention = inputs.attention_mask.to(self.device)
        features = self.whisper._mask_input_features(features, attention_mask=attention)
        whisper = self.whisper.encoder(
            features,
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state.float()
        whisper = whisper[:, : waves_16k.size(-1) // 320 + 1]
        return contentvec, whisper
