"""
AudioTextToMouth 配置（独立于 demo.project.config）：
- 维度定义（FLAME / 口型 106 维）
- 模型隐藏维度
- 训练超参与 checkpoint 目录
"""

from pathlib import Path

# Project 根目录
PROJECT_ROOT = Path(__file__).resolve().parent

# FLAME / 口型维度
N_SHAPE = 300
N_EXP = 100
N_JAW = 3
N_GLOBAL_POSE = 3
OUTPUT_DIM = N_EXP + N_JAW + N_GLOBAL_POSE  # 106

# 模型维度
Z_ID_DIM = 128
Z_INIT_DIM = 64
Z_TXT_DIM = 64
AUDIO_FEAT_DIM = 768  # Wav2Vec2 base hidden_size
FUSION_DIM = 256
DECODER_LAYERS = 4
DECODER_NHEAD = 4
# 输出头前的中间 MLP：[] 表示单层 Linear；如 [512, 512] 表示中间两层
OUTPUT_HIDDEN_LAYERS: list[int] = []

# 时序 / 音频
TARGET_FPS = 30
AUDIO_SAMPLE_RATE = 16000  # Wav2Vec2 期望 16k

# 训练超参
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 50
TRAIN_RATIO = 0.8

# checkpoint 默认目录（放在 Project 下）
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "audio_text_to_mouth"

