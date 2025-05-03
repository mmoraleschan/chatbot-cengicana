import os
import json
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import Document

# ✅ Lee la API key directamente del entorno (Render la gestiona así)
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("❌ No se encontró OPENAI_API_KEY en las variables de entorno.")

# 📁 Directorios
chunks_dir = "chunks"
vectordb_dir = "vectorstore"

# 🔍 Buscar todos los archivos .json de chunks
chunk_files = [f for f in os.listdir(chunks_dir) if f.endswith(".json")]

all_chunks = []
for filename in chunk_files:
    filepath = os.path.join(chunks_dir, filename)
    with open(filepath, "r", encoding="utf-8") as file:
        chunks = json.load(file)
        for chunk in chunks:
            if isinstance(chunk, dict) and "text" in chunk and "metadata" in chunk:
                doc = Document(
                    page_content=chunk["text"],
                    metadata=chunk["metadata"]
                )
                all_chunks.append(doc)

print(f"📦 Procesando {len(all_chunks)} chunks para vectorización...")

# 🧠 Crear embeddings
embedding = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=openai_api_key
)

# 🧠 Crear base de datos vectorial
vectordb = Chroma.from_documents(documents=all_chunks, embedding=embedding, persist_directory=vectordb_dir)
vectordb.persist()

print("✅ Base de datos vectorial actualizada exitosamente.")
