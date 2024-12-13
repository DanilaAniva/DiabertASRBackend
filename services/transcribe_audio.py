import asyncio
from functools import partial
import os
import tempfile
import subprocess
import librosa
import numpy as np

async def transcribe_audio(file_content, pipe, semaphore, executor):
    """
    Транскрибирует аудиофайл.

    Аргументы:
    - file_content: bytes, содержимое аудиофайла.
    - pipe: transformers.pipeline, модель для транскрибации.
    - semaphore: asyncio.Semaphore, для ограничения количества одновременных задач.
    - executor: ThreadPoolExecutor, для выполнения блокирующих операций.

    Возвращает:
    - dict, содержащий распознанный текст.
    """
    try:
        # Создаем временный файл для обработки
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        try:
            # Загружаем аудиофайл
            audio_array, sampling_rate = librosa.load(temp_file_path, sr=16000, mono=True)
        finally:
            # Удаляем временный файл
            os.remove(temp_file_path)
        
        # Убедимся, что audio_array это numpy array с типом float32
        audio_array = np.array(audio_array, dtype=np.float32)
        
        # Определяем синхронную функцию для обработки модели
        def process(audio):
            return pipe(
                audio,
                return_timestamps=True,
                generate_kwargs={"task": "transcribe"}
            )
        
        # Используем семафор для ограничения количества одновременных задач
        async with semaphore:
            loop = asyncio.get_running_loop()
            # Запускаем блокирующую функцию в пуле потоков
            result = await loop.run_in_executor(executor, partial(process, audio_array))
        
        return {"text": result["text"]}
    
    except subprocess.CalledProcessError as e:
        return {"error": f"Error converting audio file: {str(e)}"}
    except Exception as e:
        return {"error": str(e)}