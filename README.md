# RAG-бот для QuantumForge Software

Проектная работа по созданию RAG-бота на основе технологий Retrieval-Augmented Generation (RAG) для корпоративной базы знаний.

## Структура проекта

```
rag/
├── knowledge_base/     # База знаний (30+ документов)
├── faiss_index/      # Векторный индекс FAISS
├── scripts/        # Вспомогательные скрипты
│  ├── download_starwars_data.py # Скачивание данных
│  ├── build_index_simple.py   # Создание векторного индекса
│  ├── test_bot.py        # Тестирование бота
│  └── create_malicious_file.py  # Создание тестового файла
├── rag_bot.py       # Основной класс RAG-бота
├── app.py         # FastAPI приложение
├── requirements.txt    # Зависимости Python
├── Dockerfile       # Docker образ
├── docker-compose.yml   # Docker Compose конфигурация
└── Project_template.md   # Описание решений
```

## Установка и запуск

### 1. Настройка окружения

```bash
python -m venv .venv
source .venv/bin/activate # Linux/Mac
# или
.venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### 2. Подготовка базы знаний (Задание 2)

```bash
cd scripts
python download_starwars_data.py
```

Это создаст базу знаний с 30+ документами в директории `knowledge_base/`.

### 3. Создание векторного индекса (Задание 3)

```bash
cd scripts
python scripts/build_index_simple.py
```

Индекс будет сохранен в директории `faiss_index/`.

### 4. Запуск бота

#### Вариант 1: Консольный режим

```bash
python rag_bot.py
```

#### Вариант 2: REST API

```bash
.venv/bin/python app.py
```

Или через uvicorn:
```bash
.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

API будет доступен по адресу `http://localhost:8000`

**Swagger UI (интерактивная документация):** `http://localhost:8000/docs`

В Swagger UI вы можете:
- Просмотреть все доступные endpoints
- Протестировать API прямо в браузере
- Отправить POST запрос на `/ask` с вопросом

### 5. Docker

```bash
docker-compose up --build
```

## Тестирование

### Тестирование защиты от промпт-инъекций (Задание 5)

```bash
# Создать тестовый файл
cd scripts
python create_malicious_file.py

# Пересоздать индекс (включив тестовый файл)
python scripts/build_index_simple.py

# Запустить тесты
python test_bot.py
```

## Технологии

- **LLM**: Локальные модели через Hugging Face (или OpenAI API)
- **Эмбеддинги**: sentence-transformers/all-MiniLM-L6-v2
- **Векторная БД**: FAISS
- **Фреймворк**: LangChain
- **API**: FastAPI

## Задания

- Задание 1: Исследование моделей и инфраструктуры
- Задание 2: Подготовка базы знаний
- Задание 3: Создание векторного индекса
- Задание 4: Реализация RAG-бота с Few-shot и Chain-of-Thought
- Задание 5: Запуск и демонстрация работы бота (защита от промпт-инъекций)

## Документация

Подробное описание решений находится в файле `Project_template.md`.
