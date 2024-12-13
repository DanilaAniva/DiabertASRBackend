import transformers
import torch
import os

def init_vikhr_model():
    """
    Инициализирует модель Vikhr и возвращает pipeline для выполнения инференса.

    Возвращает:
    - pipeline: transformers.pipeline для выполнения инференса.
    - terminators: list, список терминальных токенов.
    """

    model_id = "Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={
            "torch_dtype": torch.float16,
            "quantization_config": {"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.float16},
            "low_cpu_mem_usage": True,
        },
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    return pipeline, terminators