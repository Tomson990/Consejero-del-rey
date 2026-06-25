"""
Pipeline de extracción para el proyecto Consejero del Rey.
Extrae capítulos de un epub, genera embeddings con Voyage AI,
y carga todo a Supabase (tablas: sources, chunks).

Uso:
    python extract_book.py --file "ruta/al/libro.epub" --author "Robert Greene" --title "Las 48 leyes del poder" --era moderno
"""

import os
import re
import argparse
import time
import zipfile
from bs4 import BeautifulSoup
import psycopg2
import voyageai
import anthropic
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ["SUPABASE_DB_URL"]  # connection string del pooler (puerto 6543)
VOYAGE_API_KEY = os.environ["VOYAGE_API_KEY"]

vo = voyageai.Client(api_key=VOYAGE_API_KEY)
anthropic_client = anthropic.Anthropic()  # usa ANTHROPIC_API_KEY del entorno

MIN_CHUNK_CHARS = 200   # ignorar fragmentos triviales (portada, dedicatoria, etc.)
MAX_CHUNK_CHARS = 6000  # si un capítulo es muy largo, lo subdividimos


def translate_to_spanish(text, retries=3):
    """Traduce un fragmento de texto al español usando Claude, preservando el sentido y tono."""
    for attempt in range(retries):
        try:
            response = anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4000,
                messages=[{
                    "role": "user",
                    "content": (
                        "Traducí el siguiente texto al español de forma fiel y natural, "
                        "preservando el tono y la estructura de párrafos. "
                        "Respondé solo con la traducción, sin comentarios ni preámbulo:\n\n"
                        f"{text}"
                    ),
                }],
            )
            return response.content[0].text
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  (reintentando traducción, intento {attempt + 1} falló: {e})")
            time.sleep(2)


def extract_chapters_epub(filepath):
    """
    Devuelve lista de dicts: {chapter_title, content, order}.
    Lee el epub como ZIP directo (en vez de ebooklib) para tolerar
    epubs con referencias a imágenes u otros archivos faltantes,
    que hacen fallar a ebooklib.read_epub aunque el texto esté intacto.
    """
    chapters = []

    with zipfile.ZipFile(filepath, "r") as zf:
        names = zf.namelist()
        # Archivos de contenido textual típicos de un epub
        html_names = [
            n for n in names
            if n.lower().endswith((".xhtml", ".html", ".htm"))
            and "nav" not in n.lower()  # saltar el archivo de navegación/TOC
        ]
        html_names.sort()  # orden alfabético suele coincidir con el orden de lectura

        for order, name in enumerate(html_names):
            try:
                raw = zf.read(name)
            except KeyError:
                continue  # archivo listado pero faltante; lo ignoramos y seguimos

            soup = BeautifulSoup(raw, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            text = re.sub(r"\n{3,}", "\n\n", text)

            if len(text) < MIN_CHUNK_CHARS:
                continue

            header = soup.find(["h1", "h2", "h3"])
            title = header.get_text(strip=True) if header else os.path.basename(name)

            chapters.append({
                "chapter_title": title,
                "content": text,
                "order": order,
            })

    return chapters


def extract_chapters_pdf(filepath):
    """
    Devuelve lista de dicts: {chapter_title, content, order}.
    Un PDF no trae capítulos marcados como el epub, así que se extrae
    el texto corrido y se etiqueta cada chunk con el rango de páginas
    de origen (útil para trazabilidad), dejando la subdivisión final
    en manos de split_long_chapter (misma lógica que para epub).
    """
    reader = PdfReader(filepath)
    full_text = ""
    page_count = len(reader.pages)

    for page in reader.pages:
        page_text = page.extract_text() or ""
        full_text += page_text + "\n\n"

    full_text = re.sub(r"\n{3,}", "\n\n", full_text).strip()

    # Se devuelve como un único "capítulo" grande; split_long_chapter
    # se encarga de partirlo en trozos manejables más adelante.
    return [{
        "chapter_title": f"(documento completo, {page_count} páginas)",
        "content": full_text,
        "order": 0,
    }]


def extract_chapters(filepath):
    """Despacha al extractor correcto según la extensión del archivo."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return extract_chapters_pdf(filepath)
    elif ext in (".epub",):
        return extract_chapters_epub(filepath)
    else:
        raise ValueError(f"Formato no soportado: {ext}")


def split_long_chapter(chapter):
    """Si un capítulo supera MAX_CHUNK_CHARS, lo divide por párrafos manteniendo coherencia."""
    content = chapter["content"]
    if len(content) <= MAX_CHUNK_CHARS:
        return [chapter]

    paragraphs = content.split("\n\n")
    parts = []
    current = ""
    part_num = 1

    for p in paragraphs:
        if len(current) + len(p) > MAX_CHUNK_CHARS and current:
            parts.append({
                "chapter_title": f"{chapter['chapter_title']} (parte {part_num})",
                "content": current.strip(),
                "order": chapter["order"],
            })
            part_num += 1
            current = p
        else:
            current += "\n\n" + p if current else p

    if current.strip():
        parts.append({
            "chapter_title": f"{chapter['chapter_title']} (parte {part_num})",
            "content": current.strip(),
            "order": chapter["order"],
        })

    return parts


def embed_texts(texts, batch_size=8):
    """Genera embeddings en batches usando Voyage AI (modelo voyage-3)."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        result = vo.embed(batch, model="voyage-3", input_type="document")
        all_embeddings.extend(result.embeddings)
        time.sleep(0.2)  # margen prudente para rate limits
    return all_embeddings


def load_to_supabase(title, author, era, chunks_with_embeddings):
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    cur.execute(
        "insert into sources (title, author, era) values (%s, %s, %s) returning id",
        (title, author, era),
    )
    source_id = cur.fetchone()[0]

    for idx, chunk in enumerate(chunks_with_embeddings):
        cur.execute(
            """insert into chunks (source_id, chapter_title, chunk_order, content, embedding)
               values (%s, %s, %s, %s, %s)""",
            (source_id, chunk["chapter_title"], idx, chunk["content"], chunk["embedding"]),
        )

    conn.commit()
    cur.close()
    conn.close()
    return source_id


def process_book(filepath, title, author, era, translate=False):
    print(f"Extrayendo capítulos de: {title}...")
    chapters = extract_chapters(filepath)
    print(f"  {len(chapters)} capítulos/secciones detectados (antes de subdividir largos)")

    final_chunks = []
    for ch in chapters:
        final_chunks.extend(split_long_chapter(ch))
    print(f"  {len(final_chunks)} chunks finales tras subdividir capítulos largos")

    if translate:
        print("Traduciendo chunks al español con Claude (esto puede tardar)...")
        for i, chunk in enumerate(final_chunks):
            chunk["content"] = translate_to_spanish(chunk["content"])
            print(f"  traducido {i + 1}/{len(final_chunks)}")
            time.sleep(0.3)

    print("Generando embeddings con Voyage AI...")
    texts = [c["content"] for c in final_chunks]
    embeddings = embed_texts(texts)

    for chunk, emb in zip(final_chunks, embeddings):
        chunk["embedding"] = emb

    print("Cargando a Supabase...")
    source_id = load_to_supabase(title, author, era, final_chunks)
    print(f"Listo. source_id={source_id}, {len(final_chunks)} chunks cargados.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--author", required=True)
    parser.add_argument("--era", default="moderno", choices=["clasico", "moderno"])
    parser.add_argument("--translate", action="store_true", help="Traducir el contenido al español antes de embeber")
    args = parser.parse_args()

    process_book(args.file, args.title, args.author, args.era, translate=args.translate)
