import os
import json
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

CHUNKS_DIR = "chunks"
DOCS_DIR = "docs"

def save_chunks(file_name, chunks):
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    output_path = os.path.join(CHUNKS_DIR, f"{file_name}.json")
    with open(output_path, "w") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

def process_pdf(file_path):
    try:
        loader = UnstructuredPDFLoader(file_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        return [{"text": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks]
    except Exception as e:
        print(f"❌ Error procesando {file_path}: {e}")
        return []

def process_excel(file_path):
    try:
        loader = UnstructuredExcelLoader(file_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        return [{"text": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks]
    except Exception as e:
        print(f"❌ Error procesando {file_path}: {e}")
        return []

def main():
    print("🔍 Iniciando extracción de chunks...\n")
    for file in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, file)
        if file.startswith(".") or not os.path.isfile(path):
            print(f"⚠️ Formato no soportado: {path}")
            continue

        filename_no_ext = os.path.splitext(file)[0]

        if file.lower().endswith(".pdf"):
            print(f"📄 Procesando PDF: {path}")
            chunks = process_pdf(path)
        elif file.lower().endswith(".xlsx"):
            print(f"📊 Procesando Excel: {path}")
            chunks = process_excel(path)
        else:
            print(f"⚠️ Formato no reconocido: {file}")
            continue

        if chunks:
            save_chunks(filename_no_ext, chunks)
            print(f"✅ {len(chunks)} chunks guardados en chunks/{filename_no_ext}.json\n")

if __name__ == "__main__":
    main()
