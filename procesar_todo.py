import os
import subprocess

def run_script(script_name):
    try:
        subprocess.run(["python3", script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al ejecutar {script_name}. Proceso detenido.")
        return False
    return True

def main():
    print("🚀 Iniciando procesamiento completo...\n")

    # Paso 1: Extraer chunks de documentos
    print("🔍 Paso 1: Ejecutando extraer_chunks.py...")
    if not run_script("extraer_chunks.py"):
        return

    # Paso 2: Crear vectores para base vectorial
    print("📚 Paso 2: Ejecutando crear_vectores.py...")
    if not run_script("crear_vectores.py"):
        return

    print("🎉 ¡Procesamiento completado con éxito!")

if __name__ == "__main__":
    main()
