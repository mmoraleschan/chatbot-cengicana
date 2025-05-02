import streamlit as st
import os
import time
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# === CONFIGURACIÓN DE API KEY ===
openai_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("No se encontró la API key de OpenAI")
os.environ["OPENAI_API_KEY"] = openai_key

# === CONFIGURACIÓN DE ASSISTANT ID ===
ASSISTANT_ID = "asst_Ht8bf15YVOoQizUPNbifiHf5"  # ← Reemplaza por tu ID real

# === CONFIGURACIÓN DE LA INTERFAZ ===
st.set_page_config(page_title="Asistente Técnico CENGICAÑA", page_icon="🌱", layout="wide")

# Logo y título
col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.image("logo_cengicana.png", width=80)
with col2:
    st.title("Asistente Técnico CENGICAÑA")

st.markdown("Consulta técnica sobre variedades, rendimiento y zafras. Prototipo basado en IA.")
st.markdown("---")

# === CARGA DE BASE VECTORIAL ===
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(persist_directory="vectores", embedding_function=embeddings)

# === UI DE CONSULTA ===
query = st.text_input("🔍 Escribe tu consulta:", placeholder="Ej. ¿Cuál fue la variedad más sembrada en la zafra 2024-25?")

if query:
    with st.spinner("🧠 Buscando información..."):
        # Buscar contexto en la base vectorial
        docs = vectordb.similarity_search(query, k=3)
        contexto = "\n\n".join([doc.page_content for doc in docs])

        # Crear thread con Assistant
        client = OpenAI()
        thread = client.beta.threads.create()

        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"""Responde usando únicamente el siguiente contexto técnico:

=== CONTEXTO ===
{contexto}

=== PREGUNTA ===
{query}
"""
        )

        run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=ASSISTANT_ID)

        # Esperar respuesta
        while True:
            status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if status.status == "completed":
                break
            time.sleep(1)

        messages = client.beta.threads.messages.list(thread_id=thread.id)
        respuesta = messages.data[0].content[0].text.value

    # Mostrar resultado
    st.success("✅ Respuesta del asistente:")
    st.write(respuesta)

    with st.expander("📄 Ver fragmentos utilizados"):
        for doc in docs:
            st.write(doc.page_content)

# === FOOTER ===
st.markdown("---")
st.markdown("<center>Desarrollado por mmorales como prototipo para CENGICAÑA • Versión demo • Streamlit + OpenAI</center>", unsafe_allow_html=True)