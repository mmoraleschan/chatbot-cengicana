from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Cargar PDF
loader = PyMuPDFLoader("Variedades cosechadas zafra 2021_22_23.pdf")
pages = loader.load()

# Dividir en fragmentos manejables (para futura vectorización)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(pages)

# Mostrar los primeros fragmentos extraídos
for i, doc in enumerate(docs[:3]):
    print(f"\n--- Fragmento {i+1} ---\n")
    print(doc.page_content)
