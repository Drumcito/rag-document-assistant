"""
ingesta.py
----------
Prepara la base de conocimiento. Este script se corre UNA vez (y cada que
cambien los documentos). Hace 4 cosas:

  1. Lee todos los documentos de la carpeta 'documentos/'.
  2. Los parte en fragmentos ('chunks') pequeños.
  3. Convierte cada fragmento en un vector (embedding) con Gemini.
  4. Guarda un índice FAISS y los fragmentos en disco para que rag.py los use.

Correr:  python ingesta.py
"""

import json
import os
import glob

import faiss
import numpy as np

from comun import generar_embeddings

CARPETA_DOCS = "documentos"
ARCHIVO_INDICE = "indice.faiss"
ARCHIVO_FRAGMENTOS = "fragmentos.json"

# Tamaño de cada fragmento (en caracteres) y cuánto se traslapan entre sí.
# El traslape evita cortar una idea justo a la mitad entre dos fragmentos.
TAM_FRAGMENTO = 800
TRASLAPE = 150


def partir_en_fragmentos(texto: str) -> list[str]:
    """Parte un texto largo en trozos de ~TAM_FRAGMENTO caracteres con traslape."""
    fragmentos = []
    inicio = 0
    while inicio < len(texto):
        fin = inicio + TAM_FRAGMENTO
        fragmento = texto[inicio:fin].strip()
        if fragmento:
            fragmentos.append(fragmento)
        inicio += TAM_FRAGMENTO - TRASLAPE
    return fragmentos


def leer_documentos() -> list[dict]:
    """Lee los .md y .txt de la carpeta de documentos y los parte en fragmentos."""
    rutas = glob.glob(os.path.join(CARPETA_DOCS, "*.md")) + \
            glob.glob(os.path.join(CARPETA_DOCS, "*.txt"))

    if not rutas:
        raise RuntimeError(
            f"No hay documentos en la carpeta '{CARPETA_DOCS}/'. "
            "Agrega al menos un archivo .md o .txt."
        )

    fragmentos = []
    for ruta in rutas:
        nombre = os.path.basename(ruta)
        with open(ruta, encoding="utf-8") as f:
            contenido = f.read()
        for trozo in partir_en_fragmentos(contenido):
            # Guardamos de qué archivo salió cada fragmento (la 'fuente'),
            # para poder citar de dónde viene la respuesta.
            fragmentos.append({"texto": trozo, "fuente": nombre})
    return fragmentos


def main():
    print("1/3  Leyendo y partiendo documentos...")
    fragmentos = leer_documentos()
    print(f"      {len(fragmentos)} fragmentos encontrados.")

    print("2/3  Generando embeddings con Gemini...")
    textos = [f["texto"] for f in fragmentos]
    vectores = generar_embeddings(textos)

    print("3/3  Construyendo y guardando el índice FAISS...")
    matriz = np.array(vectores, dtype="float32")
    dimension = matriz.shape[1]          # 768 para text-embedding-004
    indice = faiss.IndexFlatL2(dimension)  # búsqueda por distancia (cercanía)
    indice.add(matriz)

    faiss.write_index(indice, ARCHIVO_INDICE)
    with open(ARCHIVO_FRAGMENTOS, "w", encoding="utf-8") as f:
        json.dump(fragmentos, f, ensure_ascii=False, indent=2)

    print(f"\nListo. Índice guardado en '{ARCHIVO_INDICE}' "
          f"y fragmentos en '{ARCHIVO_FRAGMENTOS}'.")
    print("Ahora puedes correr:  python probar.py   o   uvicorn app:app --reload")


if __name__ == "__main__":
    main()
