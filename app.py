"""
app.py
------
Expone el asistente como una API web con FastAPI. Esto es lo que en la vacante
llaman "construir servicios y prototipos en Python para conectar modelos con
aplicaciones".

Correr:  uvicorn app:app --reload
Luego abre en el navegador:  http://127.0.0.1:8000/docs
(FastAPI genera sola una página para probar el endpoint.)
"""

from fastapi import FastAPI
from pydantic import BaseModel

from rag import responder

app = FastAPI(title="Asistente RAG", description="Pregunta sobre documentos internos")


# Pydantic valida que lo que llega tenga el formato correcto (un campo 'pregunta').
class Consulta(BaseModel):
    pregunta: str
    k: int = 3   # cuántos fragmentos recuperar (opcional)


class Respuesta(BaseModel):
    respuesta: str
    fuentes: list[str]


@app.get("/")
def inicio():
    """Endpoint de salud: sirve para comprobar que el servidor está vivo."""
    return {"estado": "ok", "mensaje": "Usa POST /ask o abre /docs"}


@app.post("/ask", response_model=Respuesta)
def preguntar(consulta: Consulta):
    """Recibe una pregunta y devuelve la respuesta con sus fuentes."""
    return responder(consulta.pregunta, consulta.k)
