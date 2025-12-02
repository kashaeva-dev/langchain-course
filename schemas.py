from typing import List

from pydantic import BaseModel, Field

class Source(BaseModel):
    """Схема для источника данных, использованного агентом для формирования ответа."""

    url: str = Field(description="URL источника данных")


class AgentResponse(BaseModel):
    """Схема для ответа агента, содержащая текст ответа и список источников данных."""

    answer: str = Field(description="Текст ответа")
    sources: List[Source] = Field(default_factory=list, description="Список источников данных")
