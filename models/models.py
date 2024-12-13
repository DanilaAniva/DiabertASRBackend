from pydantic import BaseModel
from typing import List

#Представляет собой текст аудиозаписи, который будет передан для анализа.
class AudioText(BaseModel):
    text: str

#Модель для представления одного действия, извлеченного из аудиозаписи.
class TimeAction(BaseModel):
    start: str
    end: str
    action: str

# Представляет собой список действий, извлеченных из аудиозаписи.
class AudioAnalysisResponse(BaseModel):
    events: List[TimeAction]