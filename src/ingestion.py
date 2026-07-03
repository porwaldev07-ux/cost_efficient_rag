import os
import hashlib
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from src.config import settings


client = QdrantClient(path=settings.QDRANT_PATH)


print("Loading local free embedding model (this may take a moment on first run)...")
embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)

COLLECTION_NAME = "document_corpus"


if not client.collection_exists(collection_name=COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=settings.EMBEDDING_DIM, distance=Distance.COSINE),
    )

def generate_deterministic_id(text: str, source_file: str) -> str:
    hasher = hashlib.md5()
    hasher.update(f"{source_file}_{text}".encode('utf-8'))
    return hasher.hexdigest()

def split_text(text: str, chunk_size: int = 500, overlap: int = 50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

def ingest_directory(directory_path: str):
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt") or filename.endswith(".md"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            chunks = split_text(content, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
            points = []
            
            print(f"Processing {filename}: Split into {len(chunks)} chunks.")
            
            for index, chunk_text in enumerate(chunks):
                # GENERATE VECTOR FOR FREE LOCALLY
                vector = embedding_model.encode(chunk_text).tolist()
                
                chunk_id = generate_deterministic_id(chunk_text, filename)
                
                points.append(PointStruct(
                    id=chunk_id,
                    vector=vector,
                    payload={
                        "text": chunk_text,
                        "source": filename,
                        "chunk_index": index,
                        "file_type": filename.split(".")[-1]
                    }
                ))
            
            client.upsert(collection_name=COLLECTION_NAME, points=points)
    print("Ingestion sequence successfully completed.")

if __name__ == "__main__":
    sample_dir = "./data/sample_corpus"
    os.makedirs(sample_dir, exist_ok=True)
    with open(f"{sample_dir}/agent_security.md", "w") as f:
        f.write("Artificial Intelligence agents execute automated operations independently based on LLM routing logic. Security filters prevent cross-tenant data leaks.")
    
    ingest_directory(sample_dir)