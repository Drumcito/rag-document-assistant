"""
comun.py
--------
Funciones compartidas para hablar con Gemini: crear el cliente, generar
embeddings (convertir texto en vectores) y pedirle una respuesta al modelo.

Lo importan tanto ingesta.py como rag.py para no repetir código.
"""

import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

# Carga las variables del archivo .env (entre ellas, GEMINI_API_KEY).
load_dotenv()

_API_KEY = os.getenv("GEMINI_API_KEY")
if not _API_KEY or _API_KEY == "pega_aqui_tu_api_key":
    raise RuntimeError(
        "No encontré tu API key. Copia .env.example como .env y pega tu key real "
        "de https://aistudio.google.com/apikey"
    )

# Un único cliente de Gemini para todo el proyecto.
cliente = genai.Client(api_key=_API_KEY)

# Nombres de los modelos que usamos (ambos tienen capa gratuita).
# Si alguno diera error 404, lista los disponibles para tu key con:
#   from google import genai; [print(m.name, m.supported_actions) for m in genai.Client().models.list()]
MODELO_EMBEDDINGS = "gemini-embedding-001"   # convierte texto -> vector numérico
MODELO_CHAT = "gemini-flash-latest"          # genera las respuestas (alias al Flash más reciente)


def _con_reintentos(funcion, intentos: int = 4, espera: float = 4.0):
    """
    Ejecuta 'funcion' y, si Gemini responde que está saturado (503) o que hay
    demasiadas peticiones (429), espera y reintenta. El nivel gratuito a veces
    devuelve estos errores temporales; así la app no se cae por eso.
    """
    for intento in range(intentos):
        try:
            return funcion()
        except (genai_errors.ServerError, genai_errors.ClientError) as e:
            codigo = getattr(e, "code", None)
            es_temporal = codigo in (429, 503)
            if not es_temporal or intento == intentos - 1:
                raise
            time.sleep(espera)


def generar_embeddings(textos: list[str]) -> list[list[float]]:
    """
    Recibe una lista de textos y devuelve una lista de vectores (uno por texto).
    Un 'embedding' es una representación numérica del significado del texto:
    textos parecidos producen vectores cercanos. Eso es lo que permite 'buscar
    por significado' en vez de por palabras exactas.

    Lo hacemos texto por texto porque el modelo de embeddings procesa uno a la
    vez; para pocos documentos es más que suficiente.
    """
    vectores = []
    for texto in textos:
        resultado = _con_reintentos(
            lambda: cliente.models.embed_content(
                model=MODELO_EMBEDDINGS,
                contents=texto,
            )
        )
        vectores.append(resultado.embeddings[0].values)
    return vectores


def preguntar_al_modelo(prompt: str) -> str:
    """Le manda un prompt a Gemini y devuelve el texto de la respuesta."""
    respuesta = _con_reintentos(
        lambda: cliente.models.generate_content(
            model=MODELO_CHAT,
            contents=prompt,
        )
    )
    return respuesta.text
