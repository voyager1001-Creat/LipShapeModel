"""
AudioTextToMouth: 初始 (shape, exp, jaw, global_pose) + 音频 [ + 可选文字 ] -> (B, T, 106) 口型序列。
音频已包含「说的什么」的信息，文本分支默认关闭（use_text=False），仅用 Whisper 做对齐。
"""

import torch
import torch.nn as nn

from Project.audio_to_mouth_config import (
    N_SHAPE,
    OUTPUT_DIM,
    Z_ID_DIM,
    Z_INIT_DIM,
    Z_TXT_DIM,
    AUDIO_FEAT_DIM,
    FUSION_DIM,
    DECODER_LAYERS,
    DECODER_NHEAD,
    OUTPUT_HIDDEN_LAYERS,
    TARGET_FPS,
)


class AudioTextToMouth(nn.Module):
    """
    输入：batch 含 shape (B,300), exp (B,100), jaw (B,3), global_pose (B,3),
         raw_audio (B,1,L), samplerate, mask (B,T)。use_text=True 时 word_id (B,) 必须提供。
    输出：(B, T, 106)，T 为 batch 内 max_T。
    """

    def __init__(
        self,
        vocab_size: int,
        n_shape: int = N_SHAPE,
        init_dim: int = 106,
        z_id_dim: int = Z_ID_DIM,
        z_init_dim: int = Z_INIT_DIM,
        z_txt_dim: int = Z_TXT_DIM,
        audio_feat_dim: int = AUDIO_FEAT_DIM,
        fusion_dim: int = FUSION_DIM,
        decoder_layers: int = DECODER_LAYERS,
        decoder_nhead: int = DECODER_NHEAD,
        output_hidden_layers: list | None = None,  # None 或 [] 表示单层 Linear
        output_dim: int = OUTPUT_DIM,
        use_wav2vec: bool = True,
        wav2vec_specifier: str = "facebook/wav2vec2-base-960h",
        wav2vec_trainable: bool = False,
        wav2vec_target_fps: int = TARGET_FPS,
        wav2vec_expected_fps: int = 50,
        use_text: bool = False,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.use_wav2vec = use_wav2vec
        self.audio_feat_dim = audio_feat_dim
        self.use_text = use_text
        self.z_txt_dim = z_txt_dim

        # Init encoders
        self.shape_mlp = nn.Sequential(
            nn.Linear(n_shape, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, z_id_dim),
        )
        self.init_mlp = nn.Sequential(
            nn.Linear(init_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, z_init_dim),
        )
        if use_text:
            self.text_embed = nn.Embedding(vocab_size, z_txt_dim, padding_idx=1)
        else:
            self.text_embed = None

        # Audio encoder: optional Wav2Vec2（使用 Project 内独立实现，不依赖 inferno）
        if use_wav2vec:
            from .audio_encoder import Wav2Vec2Encoder

            self.audio_encoder = Wav2Vec2Encoder(
                wav2vec_specifier,
                trainable=wav2vec_trainable,
                with_processor=True,
                target_fps=wav2vec_target_fps,
                expected_fps=wav2vec_expected_fps,
            )
            self.audio_feat_dim = self.audio_encoder.output_feature_dim()
        else:
            self.audio_encoder = None
            # 若不用 Wav2Vec2，调用方需提供已对齐的 audio_feature (B,T,audio_feat_dim)

        # Fusion: z_id + z_init + [z_txt] + audio -> fusion_dim（无文本时 128+64+768=960）
        fusion_in = z_id_dim + z_init_dim + self.audio_feat_dim
        if use_text:
            fusion_in += z_txt_dim
        self.fusion_proj = nn.Linear(fusion_in, fusion_dim)

        # Decoder: Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=decoder_nhead,
            dim_feedforward=fusion_dim * 2,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=decoder_layers)
        # 输出头：可带中间层 fusion_dim → h1 → h2 → ... → output_dim
        if output_hidden_layers is None:
            output_hidden_layers = OUTPUT_HIDDEN_LAYERS
        if not output_hidden_layers or output_hidden_layers == [0]:
            self.output_head = nn.Linear(fusion_dim, output_dim)
        else:
            layers = []
            d = fusion_dim
            for h in output_hidden_layers:
                layers.append(nn.Linear(d, h))
                layers.append(nn.ReLU(inplace=True))
                d = h
            layers.append(nn.Linear(d, output_dim))
            self.output_head = nn.Sequential(*layers)

    def forward(self, batch, desired_output_length=None):
        """
        batch: dict with shape, exp, jaw, global_pose, raw_audio, samplerate, word_id, mask.
        desired_output_length: 若提供则作为音频编码的 output length（帧数 T）；否则用 mask.shape[1]。
        """
        B = batch["shape"].shape[0]
        T = batch["mask"].shape[1] if "mask" in batch else desired_output_length
        if T is None:
            raise ValueError("forward 需要 batch['mask'] 或 desired_output_length 以确定序列长度 T")
        if self.use_text and "word_id" not in batch:
            raise ValueError("use_text=True 时 batch 必须包含 word_id")
        device = batch["shape"].device

        # (B, 300) -> (B, z_id_dim)
        z_id = self.shape_mlp(batch["shape"])
        # (B, 106)
        init_cat = torch.cat([batch["exp"], batch["jaw"], batch["global_pose"]], dim=1)
        z_init = self.init_mlp(init_cat)
        # 文本分支：训练时启用，word_id 必须提供
        if self.use_text:
            z_txt = self.text_embed(batch["word_id"])  # (B, z_txt_dim)
            z_txt_exp = z_txt.unsqueeze(1).expand(B, T, -1)
        else:
            z_txt_exp = None

        # Audio: (B, T, audio_feat_dim)
        if self.audio_encoder is not None:
            sample = {
                "raw_audio": batch["raw_audio"].to(device),
                "samplerate": batch["samplerate"],
            }
            out_len = desired_output_length if desired_output_length is not None else T
            self.audio_encoder(sample, desired_output_length=out_len)
            audio_feat = sample["audio_feature"]
            if audio_feat.shape[1] != T:
                audio_feat = torch.nn.functional.interpolate(
                    audio_feat.transpose(1, 2),
                    size=T,
                    mode="linear",
                    align_corners=True,
                ).transpose(1, 2)
        else:
            audio_feat = batch["audio_feature"]

        # Expand to (B, T, *) and concat（无文本时不拼 z_txt）
        z_id_exp = z_id.unsqueeze(1).expand(B, T, -1)
        z_init_exp = z_init.unsqueeze(1).expand(B, T, -1)
        if z_txt_exp is not None:
            fused = torch.cat([z_id_exp, z_init_exp, z_txt_exp, audio_feat], dim=-1)
        else:
            fused = torch.cat([z_id_exp, z_init_exp, audio_feat], dim=-1)
        fused = self.fusion_proj(fused)

        # padding mask: True = 忽略该位置（PyTorch 约定）
        key_padding_mask = (batch["mask"] < 0.5) if "mask" in batch else None
        decoded = self.decoder(fused, src_key_padding_mask=key_padding_mask)
        out = self.output_head(decoded)
        return out

