import os
import streamlit as st
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Configurar API Key
os.environ["OPENAI_API_KEY"] = "tu-api-key"
client = OpenAI()

# Assistant ID de OpenAI
ASSISTANT_ID = "asst_XXXXXX"  # <- Reemplaza por tu ID real

# Cargar base vectorial
vectordb = Chroma(
    persist_directory="vectores",
    embedding_function=OpenAIEmbeddings()
)

# Título
st.set_page_config(page_title="Asistente CENGICAÑA", layout="wide")
st.title("🤖 Asistente Técnico CENGICAÑA")

# Input del usuario
query = st.text_input("Haz una pregunta basada en el informe técnico:")

if query:
    # Buscar contexto relevante
    docs = vectordb.similarity_search(query, k=3)
    contexto = "\n\n".join([doc.page_content for doc in docs])

    # Crear nuevo thread
    thread = client.beta.threads.create()

    # Enviar mensaje con contexto
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=f"""
Basado únicamente en la siguiente información técnica responde con precisión la siguiente pregunta.

=== CONTEXTO ===
{contexto}

=== PREGUNTA ===
{query}
"""
    )

    # Ejecutar assistant
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=ASSISTANT_ID
    )

    with st.spinner("🧠 Analizando respuesta..."):
        import time
        while True:
            status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if status.status == "completed":
                break
            time.sleep(1)

    # Mostrar respuesta
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    respuesta = messages.data[0].content[0].text.value

    st.success("💡 Respuesta del asistente:")
    st.markdown(respuesta)
