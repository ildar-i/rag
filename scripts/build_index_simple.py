#!/usr/bin/env python3
"""
Упрощенный скрипт для создания векторного индекса без LangChain
"""
import os
import sys
import json
from pathlib import Path
import numpy as np

# Добавляем корневую директорию в путь
sys.path.append(str(Path(__file__).parent.parent))

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Установите зависимости: pip install faiss-cpu sentence-transformers")
    sys.exit(1)

KNOWLEDGE_BASE_DIR = "../knowledge_base"
OUTPUT_INDEX = "../faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_documents():
    """Загружает документы из базы знаний"""
    documents = []
    kb_path = Path(KNOWLEDGE_BASE_DIR)
    
    if not kb_path.exists():
        print(f"Ошибка: директория {KNOWLEDGE_BASE_DIR} не найдена")
        return documents
    
    # Загружаем все .txt файлы
    for file_path in kb_path.glob("*.txt"):
        if file_path.name == "terms_map.json":
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Извлекаем заголовок из первой строки
            lines = content.split('\n')
            title = lines[0].strip('# ').strip() if lines else file_path.stem
            
            # Остальной контент
            text_content = '\n'.join(lines[1:]).strip()
            
            if len(text_content) > 100:  # Минимальная длина
                documents.append({
                    'content': text_content,
                    'title': title,
                    'filename': file_path.name,
                    'source': str(file_path)
                })
                print(f"  ✓ Загружен: {file_path.name} ({len(text_content)} символов)")
        except Exception as e:
            print(f"  ✗ Ошибка при загрузке {file_path.name}: {e}")
    
    return documents

def split_text(text, chunk_size=1000, overlap=200):
    """Разбивает текст на чанки"""
    chunks = []
    words = text.split()
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        if len(chunk_text) > 50:  # Минимальная длина чанка
            chunks.append(chunk_text)
    
    return chunks

def create_index():
    """Создает векторный индекс"""
    print("=" * 60)
    print("Создание векторного индекса")
    print("=" * 60)
    
    # 1. Загружаем документы
    print("\n1. Загрузка документов...")
    documents = load_documents()
    
    if not documents:
        print("Ошибка: не найдено документов для индексации")
        return
    
    print(f"   Загружено документов: {len(documents)}")
    
    # 2. Разбиваем на чанки
    print("\n2. Разбиение на чанки...")
    all_chunks = []
    chunk_metadata = []
    
    for doc in documents:
        chunks = split_text(doc['content'], chunk_size=1000, overlap=200)
        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_metadata.append({
                'title': doc['title'],
                'filename': doc['filename'],
                'source': doc['source']
            })
    
    print(f"   Создано чанков: {len(all_chunks)}")
    
    # 3. Создаем эмбеддинги
    print("\n3. Создание эмбеддингов...")
    print(f"   Модель: {EMBEDDING_MODEL}")
    print("   Загрузка модели... (это может занять некоторое время)")
    
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    print("   Генерация эмбеддингов...")
    embeddings = model.encode(all_chunks, show_progress_bar=True, normalize_embeddings=True)
    
    print(f"   Размерность эмбеддингов: {embeddings.shape}")
    
    # 4. Создаем индекс FAISS
    print("\n4. Создание индекса FAISS...")
    dimension = embeddings.shape[1]
    
    # Используем L2 расстояние (Euclidean)
    index = faiss.IndexFlatL2(dimension)
    
    # Добавляем векторы в индекс
    index.add(embeddings.astype('float32'))
    
    print(f"   Векторов в индексе: {index.ntotal}")
    
    # 5. Сохраняем индекс
    print("\n5. Сохранение индекса...")
    output_path = Path(OUTPUT_INDEX)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем индекс FAISS
    faiss.write_index(index, str(output_path / "index.faiss"))
    
    # Сохраняем метаданные
    metadata = {
        "model": EMBEDDING_MODEL,
        "embedding_dim": dimension,
        "num_documents": len(documents),
        "num_chunks": len(all_chunks),
        "chunk_size": 1000,
        "chunk_overlap": 200
    }
    
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Сохраняем чанки и их метаданные
    chunks_data = {
        'chunks': all_chunks,
        'metadata': chunk_metadata
    }
    
    chunks_path = output_path / "chunks.json"
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    
    print(f"   Индекс сохранен в: {output_path}")
    print(f"   Метаданные сохранены в: {metadata_path}")
    
    # 6. Тестовый запрос
    print("\n6. Тестовый запрос...")
    query = "Что такое Synth Flux?"
    query_embedding = model.encode([query], normalize_embeddings=True).astype('float32')
    
    k = 3
    distances, indices = index.search(query_embedding, k)
    
    print(f"\n   Запрос: '{query}'")
    print(f"   Найдено результатов: {len(indices[0])}")
    if len(indices[0]) > 0:
        print(f"\n   Первый результат:")
        idx = indices[0][0]
        print(f"   Источник: {chunk_metadata[idx]['filename']}")
        print(f"   Фрагмент: {all_chunks[idx][:200]}...")
        print(f"   Расстояние: {distances[0][0]:.4f}")
    
    print("\n" + "=" * 60)
    print("Индекс успешно создан!")
    print("=" * 60)
    
    return index, all_chunks, chunk_metadata

if __name__ == "__main__":
    create_index()
