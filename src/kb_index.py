"""
Chunk the knowledge-base markdown files and build a FAISS index
for semantic retrieval.

Chunking strategy (per DATA_SCHEMA.md guidance):
- Split each file on '---' horizontal rules (major section boundaries)
- Track the nearest preceding heading as metadata
"""
import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

KB_DIR = Path(__file__).resolve().parent.parent / "knowledge-base"
INDEX_DIR = Path(__file__).resolve().parent.parent / "index"
INDEX_DIR.mkdir(exist_ok=True)

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good enough for this task


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    heading: str
    text: str


def chunk_markdown_file(path: Path) -> list[Chunk]:
    content = path.read_text(encoding="utf-8")
    sections = re.split(r"\n-{3,}\n", content)

    chunks = []
    current_heading = path.stem
    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        heading_match = re.search(r"^#+\s*(.+)$", section, re.MULTILINE)
        if heading_match:
            current_heading = heading_match.group(1).strip()
        chunks.append(Chunk(
            chunk_id=f"{path.stem}-{i}",
            source_file=str(path.relative_to(KB_DIR)),
            heading=current_heading,
            text=section,
        ))
    return chunks


def chunk_all_kb_files(kb_dir: Path = KB_DIR) -> list[Chunk]:
    all_chunks = []
    for md_file in sorted(kb_dir.rglob("*.md")):
        all_chunks.extend(chunk_markdown_file(md_file))
    return all_chunks


def build_index(chunks: list[Chunk], model: SentenceTransformer) -> faiss.Index:
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized inner product
    index.add(np.array(embeddings, dtype="float32"))
    return index


def save_index(index: faiss.Index, chunks: list[Chunk]):
    faiss.write_index(index, str(INDEX_DIR / "kb.index"))
    with open(INDEX_DIR / "kb_chunks.json", "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, indent=2)


def load_index() -> tuple[faiss.Index, list[Chunk]]:
    index = faiss.read_index(str(INDEX_DIR / "kb.index"))
    with open(INDEX_DIR / "kb_chunks.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    chunks = [Chunk(**c) for c in raw]
    return index, chunks


if __name__ == "__main__":
    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Chunking KB files...")
    chunks = chunk_all_kb_files()
    print(f"Produced {len(chunks)} chunks from KB files")

    print("Building FAISS index...")
    index = build_index(chunks, model)
    save_index(index, chunks)
    print(f"Saved index with {index.ntotal} vectors to {INDEX_DIR}")
