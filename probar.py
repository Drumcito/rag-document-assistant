"""
probar.py
---------
Forma rápida de probar el asistente desde la terminal, sin servidor web.

Correr:  python probar.py
Escribe tus preguntas; escribe 'salir' para terminar.
"""

from rag import responder


def main():
    print("Asistente RAG listo. Escribe 'salir' para terminar.\n")
    while True:
        pregunta = input("Tú > ").strip()
        if pregunta.lower() in ("salir", "exit", "quit", ""):
            break
        resultado = responder(pregunta)
        print(f"\nAsistente > {resultado['respuesta']}")
        print(f"(Fuentes: {', '.join(resultado['fuentes'])})\n")


if __name__ == "__main__":
    main()
