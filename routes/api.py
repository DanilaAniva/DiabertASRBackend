from fastapi import APIRouter, UploadFile, File, HTTPException
from models import AudioText, AudioAnalysisResponse
from services import parse_audio, transcribe_audio
import asyncio
from concurrent.futures import ThreadPoolExecutor
import torch
from executors import init_whisper_model, init_vikhr_model

router = APIRouter()

##### CONFIG #######

# 1. Максимальное количество одновременных задач
MAX_CONCURRENT_TASKS = 5
# 2. Создаём семафор
semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
# 3. Создаём пул потоков для выполнения блокирующих операций
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TASKS)
# 4. Инициализация устройства и типа данных
device = "cuda:1" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
# 5. Инициализация моделей
whisper_pipe = init_whisper_model(device, torch_dtype)
vikhr_pipeline, vikhr_terminators = init_vikhr_model()

##### ENDPOINTS #####
@router.post("/transcribe/")
async def transcribe_endpoint(file: UploadFile = File(...)):
    """
    Эндпойнт для транскрибации аудиофайла.
    Поддерживаемые форматы: mp3, wav и m4a.
    Возвращает распознанный текст.
    """
    try:
        # Проверка поддерживаемого формата
        if file.content_type not in ["audio/wav", "audio/mpeg", "audio/mp3", "audio/x-m4a", "audio/m4a"]:
            return {"error": "Unsupported file type. Please upload a .mp3, .wav or .m4a file."}
        # Чтение файла в память
        contents = await file.read()
        # Транскрибируем аудиофайл
        result = await transcribe_audio(contents, whisper_pipe, semaphore, executor)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze_audio", response_model=AudioAnalysisResponse)
async def analyze_audio_endpoint(audio_data: AudioText):
    """
    Эндпойнт для анализа аудиозаписи.
    Возвращает результаты в формате AudioAnalysisResponse.
    """
    try:
        # Обработка аудиозаписи
        result = parse_audio(audio_data, vikhr_pipeline, vikhr_terminators)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))