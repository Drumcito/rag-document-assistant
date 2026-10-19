# Asistente RAG sobre documentos (Python + Gemini + FAISS)

Asistente que responde preguntas consultando una base de documentos autorizados,
usando **RAG** (Retrieval-Augmented Generation): primero recupera los fragmentos
más relevantes y luego el modelo de lenguaje responde con base en ellos.

> Proyecto de portafolio. Demuestra: Python, integración de una **API de LLM**
> (Gemini) con manejo de JSON y autenticación por API key, **embeddings**,
> **base vectorial** (FAISS), **RAG**, salidas estructuradas y una API con **FastAPI**.

## Problema
Las personas pierden tiempo buscando en manuales y políticas internas. Este
asistente responde en lenguaje natural citando la fuente, sin inventar: si la
información no está en los documentos, lo dice.

## Cómo funciona
1. `ingesta.py` lee los documentos de `documentos/`, los parte en fragmentos y
   genera un **embedding** por fragmento (un vector que representa su significado).
2. Los vectores se guardan en un índice **FAISS** para búsqueda por similitud.
3. Al preguntar, `rag.py` convierte la pregunta en vector, recupera los fragmentos
   más cercanos y se los pasa a **Gemini** como contexto para que responda.
4. `app.py` expone todo como una API web con FastAPI.

## Tecnologías
Python · google-genai (Gemini) · FAISS · NumPy · FastAPI · Pydantic · dotenv

## Instalación y uso

```bash
# 1. Crear y activar entorno virtual (Windows PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar la API key: copia .env.example como .env y pega tu key
#    (consíguela gratis en https://aistudio.google.com/apikey)

# 4. Construir el índice a partir de los documentos
python ingesta.py

# 5a. Probar en la terminal
python probar.py

# 5b. O levantar la API y abrir http://127.0.0.1:8000/docs
uvicorn app:app --reload
```
## Resultados y limitaciones
