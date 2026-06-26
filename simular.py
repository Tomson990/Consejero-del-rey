"""
Consejero del Rey — módulo de simulación de cadenas causales.
Dado una acción concreta que estás evaluando, proyecta 3 reacciones
iniciales plausibles de la otra persona, y para cada una, 2 niveles
de consecuencias siguientes — todo anclado en patrones descritos
por los 14 autores cargados en la base.

Uso:
    python simular.py --accion "le digo a mi jefa que noté que no reconoce el proyecto"
"""

import os
import re
import json
import argparse
import psycopg2
import voyageai
import anthropic
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ["SUPABASE_DB_URL"]
VOYAGE_API_KEY = os.environ["VOYAGE_API_KEY"]

vo = voyageai.Client(api_key=VOYAGE_API_KEY)
anthropic_client = anthropic.Anthropic()

TOP_K = 10

SIMULATION_SYSTEM_PROMPT = """Sos un analista de dinámicas de poder y comportamiento humano, \
con base en los 14 autores provistos en el contexto (Maquiavelo, Sun Tzu, Gracián, Castiglione, \
Robert Greene, Cialdini, Pfeffer, Goldstein, Chase Hughes).

Dada una ACCIÓN concreta que la persona está evaluando, tu trabajo es proyectar una simulación \
de cadenas causales en forma de árbol de ramas:

1. Identificá 3 reacciones iniciales plausibles y meaningfully distintas de la otra parte \
involucrada, ancladas en los patrones de comportamiento que describen los autores del contexto.
2. Para cada una de esas 3 reacciones, proyectá 2 niveles más de consecuencias en cadena \
(qué pasa después de esa reacción, y qué pasa después de eso).

IMPORTANTE:
- No opines sobre si la acción evaluada es buena idea ni sobre el timing de actuar.
- Esto es una PROYECCIÓN DE PATRONES PLAUSIBLES según los autores, no una predicción determinista.
- Cada nodo necesita: un título de 2-4 palabras (sentence case, sin mayúsculas iniciales salvo \
nombres propios) y un subtítulo de máximo 5 palabras que lo aclare.

Respondé ÚNICAMENTE con un JSON válido, sin texto antes ni después, con esta estructura exacta:

{
  "accion": "resumen de 3-5 palabras de la acción evaluada",
  "ramas": [
    {
      "titulo": "...",
      "subtitulo": "...",
      "nivel2": {"titulo": "...", "subtitulo": "..."},
      "nivel3": {"titulo": "...", "subtitulo": "..."}
    },
    ... (exactamente 3 ramas)
  ]
}"""


def embed_query(text):
    result = vo.embed([text], model="voyage-3", input_type="query")
    return result.embeddings[0]


def search_relevant_chunks(query_embedding, top_k=TOP_K):
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute(
        """
        select c.content, c.chapter_title, s.title, s.author
        from chunks c
        join sources s on s.id = c.source_id
        order by c.embedding <=> %s::vector
        limit %s
        """,
        (query_embedding, top_k),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{"content": r[0], "chapter_title": r[1], "book_title": r[2], "author": r[3]} for r in rows]


def build_context(chunks):
    parts = [
        f"[Fuente: {c['author']} — \"{c['book_title']}\", sección: {c['chapter_title']}]\n{c['content']}"
        for c in chunks
    ]
    return "\n\n---\n\n".join(parts)


def extract_json(text):
    """Extrae el bloque JSON de la respuesta, tolerando que venga envuelto en ```json ... ```."""
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No se encontró JSON en la respuesta: {text[:200]}")
    return json.loads(match.group(0))


def simular_cadena(accion):
    print("Buscando patrones relevantes en los 14 libros...")
    query_embedding = embed_query(accion)
    chunks = search_relevant_chunks(query_embedding)
    context = build_context(chunks)

    print("Generando árbol de simulación...\n")
    user_message = (
        f"CONTEXTO DE LOS LIBROS:\n\n{context}\n\n---\n\nACCIÓN A EVALUAR:\n{accion}"
    )

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1200,
        system=SIMULATION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw_text = response.content[0].text
    tree = extract_json(raw_text)
    return tree


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--accion", required=True, help="La acción concreta que estás evaluando")
    args = parser.parse_args()

    tree = simular_cadena(args.accion)
    print(json.dumps(tree, indent=2, ensure_ascii=False))
