"""
音频词级对齐：Whisper 转写返回 (word, start_sec, end_sec) 列表。
从 demo/align_audio_words_to_frames 抽离，供 Project 独立使用。
"""
from pathlib import Path

# ModelScope 上的 faster-whisper 模型 ID
MODELSCOPE_WHISPER_IDS = {
    "base": "pengzhendong/faster-whisper-base",
    "small": "pengzhendong/faster-whisper-small",
    "medium": "pengzhendong/faster-whisper-medium",
    "large": "pengzhendong/faster-whisper-large",
    "large-v2": "pengzhendong/faster-whisper-large-v2",
    "large-v3": "pengzhendong/faster-whisper-large-v3",
}

_WHISPER_CACHE_SUBDIR = {
    "base": "faster-whisper-base",
    "small": "faster-whisper-small",
    "medium": "faster-whisper-medium",
    "large": "faster-whisper-large",
    "large-v2": "faster-whisper-large-v2",
    "large-v3": "faster-whisper-large-v3",
}


def _get_whisper_model_path(
    model_size: str, use_modelscope: bool, cache_dir: Path
) -> str:
    """解析为 faster-whisper 可用的模型路径。"""
    path = Path(model_size)
    if path.exists() and (path / "model.bin").is_file():
        return str(path.resolve())
    if path.exists() and path.is_dir():
        return str(path.resolve())
    if model_size in _WHISPER_CACHE_SUBDIR:
        subdir = _WHISPER_CACHE_SUBDIR[model_size]
        for prefix in ("pengzhendong", "Systran"):
            candidate = cache_dir / prefix / subdir
            if (candidate / "model.bin").is_file():
                return str(candidate.resolve())
        flat = cache_dir / subdir
        if (flat / "model.bin").is_file():
            return str(flat.resolve())
    if use_modelscope and model_size in MODELSCOPE_WHISPER_IDS:
        try:
            from modelscope.hub.snapshot_download import snapshot_download

            model_id = MODELSCOPE_WHISPER_IDS[model_size]
            cache_dir.mkdir(parents=True, exist_ok=True)
            local_dir = snapshot_download(model_id=model_id, cache_dir=cache_dir)
            return local_dir
        except ImportError:
            raise ImportError(
                "ModelScope 下载需安装: pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple"
            )
    return model_size


def run_whisper(
    audio_path: Path,
    model_size: str = "base",
    no_ssl_verify: bool = False,
    use_modelscope: bool = False,
    cache_dir: Path = None,
    device: str = "cpu",
):
    """用 Whisper 转写（优先 faster-whisper，否则 openai-whisper），返回 (word, start_sec, end_sec) 列表。"""
    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parent / "model" / "whisper_models"
    if no_ssl_verify:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
    try:
        from faster_whisper import WhisperModel

        model_path = _get_whisper_model_path(model_size, use_modelscope, cache_dir)
        compute_type = "int8" if device == "cpu" else "float16"
        model = WhisperModel(model_path, device=device, compute_type=compute_type)
        segments, _ = model.transcribe(str(audio_path), word_timestamps=True)
        words = []
        for seg in segments:
            if not getattr(seg, "words", None):
                continue
            for w in seg.words:
                words.append((w.word.strip(), float(w.start), float(w.end)))
        return words
    except ImportError:
        pass

    try:
        import whisper

        model = whisper.load_model(model_size)
        out = model.transcribe(str(audio_path), word_timestamps=True)
        words = []
        for seg in out.get("segments") or []:
            seg_words = seg.get("words")
            if seg_words:
                for w in seg_words:
                    words.append(
                        (
                            w.get("word", "").strip(),
                            float(w.get("start", 0)),
                            float(w.get("end", 0)),
                        )
                    )
            elif seg.get("text"):
                words.append(
                    (
                        seg["text"].strip(),
                        float(seg.get("start", 0)),
                        float(seg.get("end", 0)),
                    )
                )
        return words
    except ImportError:
        raise ImportError(
            "Need either: pip install faster-whisper   or   pip install openai-whisper"
        )
