#!/usr/bin/env python3
"""
FastAPI приложение для RAG-бота
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from rag_bot_simple import RAGBot
from pathlib import Path

app = FastAPI(
    title="RAG Bot API", 
    description="API для RAG-бота QuantumForge Software",
    version="1.0.0"
)

# Глобальная переменная для бота
bot: Optional[RAGBot] = None

class QuestionRequest(BaseModel):
    question: str
    k: int = 3
    use_filter: bool = True
    relevance_threshold: float = 1.4

class SourceInfo(BaseModel):
    filename: str
    title: str
    source: Optional[str] = None

class QuestionResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    reasoning: str

@app.on_event("startup")
async def startup_event():
    """Инициализация бота при старте приложения"""
    global bot
    try:
        bot = RAGBot()
        print("Бот успешно инициализирован")
    except Exception as e:
        print(f"Ошибка при инициализации бота: {e}")
        raise

@app.get("/")
async def root():
    """Корневой endpoint"""
    return {
        "message": "RAG Bot API",
        "status": "running",
        "endpoints": {
            "ask": "/ask - Задать вопрос боту",
            "health": "/health - Проверка здоровья сервиса"
        }
    }

@app.get("/health")
async def health():
    """Проверка здоровья сервиса"""
    return {
        "status": "healthy" if bot else "not_initialized",
        "index_exists": Path("./faiss_index").exists()
    }

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Задать вопрос боту"""
    if not bot:
        raise HTTPException(status_code=503, detail="Бот не инициализирован")
    
    try:
        result = bot.ask(
            question=request.question,
            k=request.k,
            use_filter=request.use_filter,
            relevance_threshold=request.relevance_threshold
        )
        # Преобразуем источники в правильный формат
        sources = [SourceInfo(**src) for src in result['sources']]
        return QuestionResponse(
            answer=result['answer'],
            sources=sources,
            reasoning=result['reasoning']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке вопроса: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Используем порт 8001, так как 8000 может быть занят
    uvicorn.run(app, host="0.0.0.0", port=8001)
