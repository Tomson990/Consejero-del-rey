"""
Consejero del Rey — interfaz Streamlit.
Misma lógica que consultar.py, envuelta en una interfaz web simple.
"""

import streamlit as st
import psycopg2
import voyageai
import anthropic

st.set_page_config(page_title="Consejero del Rey", page_icon="♟️", layout="centered")

DATABASE_URL = st.secrets["SUPABASE_DB_URL"]
VOYAGE_API_KEY = st.secrets["VOYAGE_API_KEY"]

vo = voyageai.Client(api_key=VOYAGE_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

TOP_K = 12

SYSTEM_PROMPT = """Sos un consejero personal con la sabiduría combinada de consejeros de \
corte antiguos (Maquiavelo, Sun Tzu, Gracián, Castiglione) y analistas modernos del poder, \
la persuasión y el comportamiento humano (Robert Greene, Cialdini, Pfeffer, Goldstein, \
Chase Hughes).

Tu trabajo, dada una situación real que te plantea la persona, es:

1. Leer la situación en términos de dinámica de poder, percepción social, e intereses en juego.
2. Citar específicamente qué dirían 2-3 de los autores/fuentes provistas en el contexto sobre \
esta situación particular — nombrándolos por nombre, no de forma genérica. Usá las ideas \
provistas en el contexto, parafraseadas en tus propias palabras, nunca como cita textual larga.
3. Dar una lectura de postura: qué conviene priorizar, qué riesgos hay en cada camino posible.

IMPORTANTE: Nunca le digas a la persona si debe actuar ahora o esperar, ni opines sobre el \
timing de su decisión. Esa parte la dejás completamente en sus manos. Tu rol es dar \
perspectiva e interpretación, no instrucciones sobre el momento de actuar.

Sé directo, sin rodeos protocolares. No hace falta que repitas la situación que te contaron."""


@st.cache_resource
def get_connection():
    return psycopg2.connect(DATABASE_URL)


def embed_query(text):
    result = vo.embed([text], model="voyage-3", input_type="query")
    return result.embeddings[0]


def search_relevant_chunks(query_embedding, top_k=TOP_K):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        select c.id, c.content, c.chapter_title, s.title, s.author, s.era,
               1 - (c.embedding <=> %s::vector) as similarity
        from chunks c
        join sources s on s.id = c.source_id
        order by c.embedding <=> %s::vector
        limit %s
        """,
        (query_embedding, query_embedding, top_k),
    )
    rows = cur.fetchall()
    cur.close()

    return [
        {
            "chunk_id": r[0], "content": r[1], "chapter_title": r[2],
            "book_title": r[3], "author": r[4], "era": r[5], "similarity": r[6],
        }
        for r in rows
    ]


def build_context(chunks):
    parts = [
        f"[Fuente: {c['author']} — \"{c['book_title']}\", sección: {c['chapter_title']}]\n{c['content']}"
        for c in chunks
    ]
    return "\n\n---\n\n".join(parts)


def ask_advisor(situation, chunks):
    context = build_context(chunks)
    user_message = (
        f"CONTEXTO DE LOS LIBROS (fragmentos relevantes encontrados):\n\n{context}\n\n"
        f"---\n\nMI SITUACIÓN:\n{situation}"
    )
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


def save_query(situation, response_text, chunks):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "insert into queries (situation_text, response_text, status) values (%s, %s, 'ready') returning id",
        (situation, response_text),
    )
    query_id = cur.fetchone()[0]
    for c in chunks:
        cur.execute(
            "insert into query_sources (query_id, chunk_id, relevance_score) values (%s, %s, %s)",
            (query_id, c["chunk_id"], c["similarity"]),
        )
    conn.commit()
    cur.close()
    return query_id


# --- Interfaz ---

st.title("♟️ Consejero del Rey")
st.caption("14 voces de poder, estrategia e influencia — antiguas y modernas — para leer tu situación.")

situation = st.text_area(
    "Contame tu situación",
    height=150,
    placeholder="Ej: Mi jefa no reconoce un proyecto que armé solo, aunque el resto del equipo sí lo valora...",
)

if st.button("Consultar", type="primary", use_container_width=True):
    if not situation.strip():
        st.warning("Escribí tu situación primero.")
    else:
        with st.spinner("Buscando entre los 14 libros..."):
            query_embedding = embed_query(situation)
            chunks = search_relevant_chunks(query_embedding)

        with st.spinner("Consultando al consejero..."):
            response_text = ask_advisor(situation, chunks)
            query_id = save_query(situation, response_text, chunks)

        st.markdown("---")
        st.markdown(response_text)

        with st.expander("Ver fuentes consultadas"):
            for c in chunks:
                st.caption(f"**{c['author']}** — *{c['book_title']}* ({c['chapter_title']}) · similitud {c['similarity']:.2f}")
