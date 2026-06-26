"""
Consejero del Rey — interfaz Streamlit.
Misma lógica que consultar.py, envuelta en una interfaz web simple.
"""

import streamlit as st
import psycopg2
import voyageai
import anthropic
import json
import re

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
- Cada nodo necesita: un título de 2-4 palabras (sentence case) y un subtítulo de máximo 5 palabras.

Respondé ÚNICAMENTE con un JSON válido, sin texto antes ni después, con esta estructura exacta:

{
  "ramas": [
    {
      "titulo": "...", "subtitulo": "...",
      "nivel2": {"titulo": "...", "subtitulo": "..."},
      "nivel3": {"titulo": "...", "subtitulo": "..."}
    },
    ... (exactamente 3 ramas)
  ]
}"""


def extract_json(text):
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No se encontró JSON en la respuesta del modelo.")
    return json.loads(match.group(0))


def validar_estructura(tree):
    """Verifica que el JSON tenga la forma esperada, con mensajes claros si falta algo."""
    if "ramas" not in tree:
        raise ValueError("la respuesta no incluye la clave 'ramas'")
    if len(tree["ramas"]) < 3:
        raise ValueError(f"se esperaban 3 ramas, llegaron {len(tree['ramas'])}")
    for i, rama in enumerate(tree["ramas"][:3]):
        for campo in ["titulo", "subtitulo", "nivel2", "nivel3"]:
            if campo not in rama:
                raise ValueError(f"a la rama {i + 1} le falta el campo '{campo}'")
        for nivel in ["nivel2", "nivel3"]:
            for campo in ["titulo", "subtitulo"]:
                if campo not in rama[nivel]:
                    raise ValueError(f"a {nivel} de la rama {i + 1} le falta '{campo}'")


def simular_cadena(accion, top_k=10, max_intentos=2):
    query_embedding = embed_query(accion)
    chunks = search_relevant_chunks(query_embedding, top_k=top_k)
    context = build_context(chunks)
    user_message = f"CONTEXTO DE LOS LIBROS:\n\n{context}\n\n---\n\nACCIÓN A EVALUAR:\n{accion}"

    last_error = None
    for intento in range(max_intentos):
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            system=SIMULATION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        raw_text = response.content[0].text
        st.session_state["ultimo_json_crudo"] = raw_text
        try:
            tree = extract_json(raw_text)
            validar_estructura(tree)
            return tree, chunks
        except (ValueError, json.JSONDecodeError) as e:
            last_error = e
            continue

    raise ValueError(f"no se pudo generar una estructura válida tras {max_intentos} intentos ({last_error})")


def esc(s):
    """Escapa caracteres especiales de XML para que no rompan el SVG."""
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;"))


def render_tree_svg(tree):
    ramas = tree["ramas"][:3]
    colors = [
        {"fill": "#FAECE7", "stroke": "#D85A30", "text": "#4A1B0C"},  # coral
        {"fill": "#FAEEDA", "stroke": "#BA7517", "text": "#412402"},  # amber
        {"fill": "#E1F5EE", "stroke": "#0F6E56", "text": "#04342C"},  # teal
    ]
    neutral = {"fill": "#F1EFE8", "stroke": "#5F5E5A", "text": "#2C2C2A"}
    gray = {"fill": "#F1EFE8", "stroke": "#5F5E5A", "text": "#2C2C2A"}

    svg_parts = [
        '<svg width="100%" viewBox="0 0 680 420" xmlns="http://www.w3.org/2000/svg" '
        'role="img" style="font-family: -apple-system, sans-serif;">',
        "<title>Simulación de cadenas causales</title>",
        "<desc>Árbol de reacciones probables y sus consecuencias en cadena</desc>",
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" '
        'markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" '
        'stroke="#888780" stroke-width="1.5" stroke-linecap="round" '
        'stroke-linejoin="round"/></marker></defs>',
    ]

    def box(x, y, w, h, c, title, subtitle):
        cx = x + w / 2
        return (
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" '
            f'fill="{c["fill"]}" stroke="{c["stroke"]}" stroke-width="1"/>'
            f'<text x="{cx}" y="{y + h * 0.4}" text-anchor="middle" dominant-baseline="central" '
            f'font-size="14" font-weight="500" fill="{c["text"]}">{esc(title)}</text>'
            f'<text x="{cx}" y="{y + h * 0.72}" text-anchor="middle" dominant-baseline="central" '
            f'font-size="12" fill="{c["text"]}">{esc(subtitle)}</text>'
        )

    centers = [140, 340, 540]

    svg_parts.append(box(240, 20, 200, 44, gray, "Acción evaluada", ""))

    for cx in centers:
        x1 = 290 if cx == 140 else (390 if cx == 540 else 340)
        svg_parts.append(
            f'<line x1="{x1}" y1="64" x2="{cx}" y2="108" stroke="#888780" stroke-width="1" marker-end="url(#arrow)"/>'
        )

    for rama, cx, color in zip(ramas, centers, colors):
        bx = cx - 100
        svg_parts.append(box(bx, 110, 200, 56, color, rama["titulo"], rama["subtitulo"]))
        svg_parts.append(f'<line x1="{cx}" y1="166" x2="{cx}" y2="198" stroke="#888780" stroke-width="1" marker-end="url(#arrow)"/>')

        n2 = rama["nivel2"]
        svg_parts.append(box(bx, 200, 200, 56, neutral, n2["titulo"], n2["subtitulo"]))
        svg_parts.append(f'<line x1="{cx}" y1="256" x2="{cx}" y2="288" stroke="#888780" stroke-width="1" marker-end="url(#arrow)"/>')

        n3 = rama["nivel3"]
        svg_parts.append(box(bx, 290, 200, 56, neutral, n3["titulo"], n3["subtitulo"]))

    svg_parts.append(
        '<text x="340" y="380" text-anchor="middle" font-size="12" fill="#5F5E5A">'
        "Proyección de patrones plausibles según los 14 autores — no es una predicción determinista</text>"
    )
    svg_parts.append("</svg>")
    return "\n".join(svg_parts)




st.title("♟️ Consejero del Rey")
st.caption("14 voces de poder, estrategia e influencia — antiguas y modernas.")

tab_consejero, tab_simulador = st.tabs(["Consejero", "Simulador de cadenas"])

with tab_consejero:
    situation = st.text_area(
        "Contame tu situación",
        height=150,
        placeholder="Ej: Mi jefa no reconoce un proyecto que armé solo, aunque el resto del equipo sí lo valora...",
        key="situation_input",
    )

    if st.button("Consultar", type="primary", use_container_width=True, key="consultar_btn"):
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

with tab_simulador:
    st.caption("Describí una acción concreta que estás evaluando, y proyectá las reacciones posibles en cadena.")
    accion = st.text_area(
        "¿Qué acción estás evaluando?",
        height=100,
        placeholder="Ej: Le digo a mi jefa que noté que no reconoce el proyecto...",
        key="accion_input",
    )

    if st.button("Simular", type="primary", use_container_width=True, key="simular_btn"):
        if not accion.strip():
            st.warning("Escribí la acción que estás evaluando primero.")
        else:
            with st.spinner("Proyectando reacciones posibles..."):
                try:
                    tree, sim_chunks = simular_cadena(accion)
                    svg_code = render_tree_svg(tree)
                    st.markdown("---")
                    st.components.v1.html(
                        f'<div style="width:100%">{svg_code}</div>',
                        height=460,
                    )
                    st.caption("Proyección de patrones plausibles según los 14 autores — no es una predicción determinista.")
                except Exception as e:
                    st.error(f"No se pudo generar la simulación: {e}")
                    if "ultimo_json_crudo" in st.session_state:
                        with st.expander("Ver JSON crudo (diagnóstico)"):
                            st.code(st.session_state["ultimo_json_crudo"], language="json")

