import os
import time
import base64
import streamlit as st
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Configurar clave desde secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
client = OpenAI()
ASSISTANT_ID = "asst_xxxxxxxxxxxxxx"  # ← Reemplaza por tu ID real

# Cargar base vectorial
vectordb = Chroma(
    persist_directory="vectores",
    embedding_function=OpenAIEmbeddings()
)

# Función para mostrar el logo
def show_logo(path, width=200):
    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f'<img src="data:image/png;base64,{encoded}" width="{width}">',
        unsafe_allow_html=True
    )

# Personalización de la app
st.set_page_config(page_title="Chatbot CENGICAÑA", layout="wide")

# Encabezado con logo y título
col1, col2 = st.columns([1, 6])
with col1:
    show_logo("logo_cengicana.png", width=120)
with col2:
    st.markdown("<h1 style='color:#3B6C26; margin-top: 15px;'>Asistente Técnico CENGICAÑA</h1>", unsafe_allow_html=True)
    st.markdown("Consulta técnica sobre variedades, rendimiento y zafras. Prototipo basado en IA.")

# Línea separadora
st.markdown("---")

# Entrada del usuario
query = st.text_input("🔍 Escribe tu consulta:", placeholder="Ej. ¿Cuál fue la variedad más sembrada en la zafra 2024-25?")

if query:
    # Buscar contexto en base vectorial
    docs = vectordb.similarity_search(query, k=3)
    contexto = "\n\n".join([d.page_content for d in docs])

    # Crear conversación con Assistant API
    thread = client.beta.threads.create()
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=f"""
Basado únicamente en la siguiente información técnica, responde de forma clara y profesional.

=== CONTEXTO ===
{contexto}

=== PREGUNTA ===
{query}
"""
    )

    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=ASSISTANT_ID)

    with st.spinner("🧠 Buscando información..."):
        while True:
            status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if status.status == "completed":
                break
            time.sleep(1)

    messages = client.beta.threads.messages.list(thread_id=thread.id)
    respuesta = messages.data[0].content[0].text.value

    # Mostrar respuesta
    st.success("✅ Respuesta del asistente:")
    st.markdown(respuesta)

    # Mostrar fragmentos usados (opcional)
    with st.expander("📄 Ver fragmentos utilizados"):
        for i, d in enumerate(docs):
            st.markdown(f"**Fragmento {i+1}:**\n{d.page_content}")

# Footer institucional
st.markdown("""
<hr style="margin-top:50px; margin-bottom:10px;">
<p style="text-align: center; font-size: 0.85em; color: #888;">
Desarrollado por mmorales como prototipo para CENGICAÑA • Versión demo • Streamlit + OpenAI
</p>
""", unsafe_allow_html=True)