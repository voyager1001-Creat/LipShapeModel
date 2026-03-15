"""
独立 Wav2Vec2 编码器：供 Project 内 AudioTextToMouth 使用，不依赖 inferno。
从 inferno.models.temporal.AudioEncoders 抽离并去掉 inferno 依赖。
"""
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2Model,
    Wav2Vec2Processor,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2BaseModelOutput


def temporal_interpolation(features, input_fps, output_fps, output_len=None):
    features = features.transpose(1, 2)
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(math.ceil(seq_len * output_fps))
    output_features = F.interpolate(
        features, size=output_len, align_corners=True, mode="linear"
    )
    return output_features.transpose(1, 2)


class Wav2Vec2ModelResampled(Wav2Vec2Model):
    """Wav2Vec2 + 时间维重采样到 target_fps。"""

    def __init__(self, config, target_fps=25, model_expected_fps=50):
        super().__init__(config)
        self.model_expected_fps = model_expected_fps
        self.target_fps = target_fps

    def forward(
        self,
        input_values,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        desired_output_length=None,
    ):
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        if self.model_expected_fps != self.target_fps or desired_output_length is not None:
            extract_features = temporal_interpolation(
                extract_features,
                self.model_expected_fps,
                self.target_fps,
                output_len=desired_output_length,
            )
        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask
            )
        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states,
            mask_time_indices=mask_time_indices,
            attention_mask=attention_mask,
        )
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = encoder_outputs[0]
        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]
        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def _get_feat_extract_output_lengths(self, input_lengths, add_adapter=None):
        input_lengths = super()._get_feat_extract_output_lengths(
            input_lengths, add_adapter
        )
        input_lengths = (
            input_lengths.to(torch.float32)
            / (self.model_expected_fps / self.target_fps)
        ).to(torch.int64)
        return input_lengths


def _resolve_wav2vec2_local(model_specifier: str) -> str:
    """若为 HF id 且 Project/model/wav2vec2-base-960h 存在，则用本地路径。"""
    if model_specifier != "facebook/wav2vec2-base-960h":
        return model_specifier
    local = Path(__file__).resolve().parent / "wav2vec2-base-960h"
    if local.is_dir() and (local / "config.json").exists():
        return str(local)
    return model_specifier


class Wav2Vec2Encoder(torch.nn.Module):
    """独立于 inferno 的 Wav2Vec2 编码器，接口与 inferno 版一致。"""

    def __init__(
        self,
        model_specifier: str,
        trainable: bool,
        with_processor: bool = True,
        target_fps: int = 25,
        expected_fps: int = 50,
        freeze_feature_extractor: bool = True,
        dropout_cfg: dict = None,
    ):
        super().__init__()
        model_specifier = _resolve_wav2vec2_local(model_specifier)
        self.model_specifier = model_specifier
        self.cfg = Wav2Vec2Config.from_pretrained(model_specifier)
        self.dropout = None  # 可选 dropout，Project 暂不传 dropout_cfg
        if with_processor:
            self.input_processor = Wav2Vec2Processor.from_pretrained(model_specifier)
        else:
            self.input_processor = None
        if not target_fps or not expected_fps:
            self.model = Wav2Vec2Model.from_pretrained(model_specifier)
            self.resampling = False
        else:
            self.model = Wav2Vec2ModelResampled.from_pretrained(model_specifier)
            self.resampling = True
            self.model.model_expected_fps = expected_fps
            self.model.target_fps = target_fps
        self.trainable = trainable
        if freeze_feature_extractor:
            self.model.feature_extractor._freeze_parameters()
        if not trainable:
            self.model.requires_grad_(False)

    def get_trainable_parameters(self):
        if self.trainable:
            return [p for p in self.model.parameters() if p.requires_grad]
        return []

    def _forward(self, sample, train=False, desired_output_length=None):
        if self.input_processor is not None:
            raw_audio = sample["raw_audio"].view(sample["raw_audio"].shape[0], -1)
            # 按「每条一条」传 list，避免 processor 把 (B,L) 当成单条多通道导致 4D
            sr = sample["samplerate"][0] if isinstance(sample["samplerate"], (list, tuple)) else sample["samplerate"]
            audio_list = [raw_audio[i].cpu().numpy() for i in range(raw_audio.shape[0])]
            proc = self.input_processor(
                audio_list,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
            )
            input_val = proc.input_values.to(device=raw_audio.device)
            if input_val.dim() == 1:
                input_val = input_val.unsqueeze(0)
            # 若 processor 误把 (B,L) 当单条多通道会返回 4D (1,1,B,L)，转为 (B, L)
            if input_val.dim() == 4:
                input_val = input_val.squeeze(0).squeeze(0)
            elif input_val.dim() == 3 and input_val.shape[0] == 1:
                input_val = input_val.squeeze(0)
        else:
            input_val = sample["processed_audio"].view(
                sample["processed_audio"].shape[0], -1
            )
        B = input_val.shape[0]
        if isinstance(self.model, Wav2Vec2ModelResampled):
            feats_ = self.model(input_val, desired_output_length=desired_output_length)
        else:
            feats_ = self.model(input_val)
        T2 = feats_.last_hidden_state.shape[1]
        audio_feat = feats_.last_hidden_state
        if desired_output_length is not None and T2 != desired_output_length:
            audio_feat = audio_feat.transpose(1, 2)
            audio_feat = F.interpolate(
                audio_feat,
                size=desired_output_length,
                mode="linear",
                align_corners=True,
            )
            audio_feat = audio_feat.transpose(1, 2)
        sample["audio_feature"] = audio_feat
        if self.dropout is not None:
            sample["audio_feature"] = self.dropout(sample["audio_feature"])
        return sample

    def train(self, mode: bool = True):
        mode = mode and self.trainable
        self.model.train(mode)
        return self

    def forward(self, sample, train=False, desired_output_length=None):
        if self.trainable:
            return self._forward(
                sample, train=train, desired_output_length=desired_output_length
            )
        with torch.no_grad():
            return self._forward(
                sample, train=train, desired_output_length=desired_output_length
            )

    def output_feature_dim(self):
        return self.cfg.hidden_size
