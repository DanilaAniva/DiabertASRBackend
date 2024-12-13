from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def init_whisper_model(device, torch_dtype):
    """
    Инициализирует модель Whisper для транскрибации аудиофайлов.

    Аргументы:
    - device: str, устройство для выполнения (например, "cuda:1" или "cpu").
    - torch_dtype: torch.dtype, тип данных для модели (например, torch.float16).

    Возвращает:
    - pipeline: transformers.pipeline, готовый к использованию пайплайн для транскрибации.
    """
    model_id = "openai/whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    return pipe