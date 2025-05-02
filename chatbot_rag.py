import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# Configura tu clave de OpenAI
os.environ["OPENAI_API_KEY"] = "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"
client = OpenAI()

# ID del assistant creado en platform.openai.com
ASSISTANT_ID = "asst_mJkaEWYpWCemkgZDWXsw9MVm"  # reemplaza por tu ID real

# Cargar la base vectorial
vectordb = Chroma(
    persist_directory="vectores",
    embedding_function=OpenAIEmbeddings()
)

# Bucle de conversación
while True:
    user_query = input("\n🔍 Escribe tu pregunta (o 'salir' para terminar):\n> ")
    if user_query.lower() == "salir":
        break

    # Paso 1: Buscar en la base vectorial (RAG)
    docs = vectordb.similarity_search(user_query, k=3)
    contexto = "\n\n".join([d.page_content for d in docs])

    # Paso 2: Crear el thread y enviar la pregunta con contexto
    thread = client.beta.threads.create()

    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=f"""
Basado únicamente en la siguiente información técnica responde con precisión la siguiente pregunta.

=== CONTEXTO ===
{contexto}

=== PREGUNTA ===
{user_query}
"""
    )

    # Paso 3: Ejecutar el assistant
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=ASSISTANT_ID
    )

    # Esperar a que termine
    import time
    while True:
        run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run_status.status == "completed":
            break
        time.sleep(1)

    # Mostrar respuesta
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    respuesta = messages.data[0].content[0].text.value
    print("\n💡 Respuesta del assistant:\n", respuesta)
