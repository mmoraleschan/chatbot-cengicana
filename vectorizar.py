from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# 1. Cargar el PDF
loader = PyMuPDFLoader("Variedades cosechadas zafra 2021_22_23.pdf")
pages = loader.load()

# 2. Dividir en fragmentos
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(pages)

# 3. Crear embeddings con OpenAI
embeddings = OpenAIEmbeddings()  # Asegúrate de tener tu API key configurada como variable de entorno

# 4. Crear base vectorial en disco
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="vectores"
)

vectordb.persist()
print("✅ Base vectorial creada con éxito.")
