import json
from tqdm import tqdm
from json_repair import repair_json
from prompts.system_prompt import system_prompt

def parse_audio(audio_text, pipeline, terminators):
    """
    Обрабатывает одну аудиозапись и возвращает результаты в формате JSON.

    Аргументы:
    - audio_text: AudioText, содержит текст аудиозаписи для обработки.
    - pipeline: transformers.pipeline, модель для инференса.
    - terminators: list, список терминальных токенов.

    Возвращает:
    - dict, результаты в формате {'events': [список TimeAction]}.
    """
    # Формируем сообщения для модели
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Аудиозапись на вход: {audio_text.text}"},
    ]
    
    # Генерируем промпт для модели
    prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Выполняем инференс
    outputs = pipeline(
        prompt,
        max_new_tokens=200,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
    )
    
    # Получаем ответ модели
    response = outputs[0]["generated_text"][len(prompt):].strip()
    
    try:
        # Попытка преобразовать ответ в JSON
        structured_response = json.loads(response)
    except json.JSONDecodeError:
        # Если JSON некорректен, исправляем его с помощью json_repair
        try:
            repaired_response = repair_json(response)  # Исправляем JSON
            structured_response = json.loads(repaired_response)  # Десериализуем исправленный JSON
        except Exception as e:
            # Если исправление не удалось, возвращаем ошибку
            raise ValueError(f"Failed to parse model response: {str(e)}")
    
    # Возвращаем результат
    return {"events": structured_response}