import pymupdf
from typing import List

from tqdm import tqdm

from core.core import (
    load_yaml_to_dict,
    LLMClient
)
from core.indexer import (
    texts2embeddings,
    BuildFaissIndex
)


def pdf_page_text_generator(pdf_path: str):
    document = pymupdf.open(pdf_path)
    for i, page in enumerate(document):
        text = page.get_text()
        yield (i, text)
    document.close()


class TextChunker:
    def __init__(self, chunk_size: int = 800, overlap: int = 80):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.overlap
        return chunks


if __name__ == "__main__":
    cfg = load_yaml_to_dict("./cfg.yaml")
    cfg_core = cfg["core"]
    cfg_indexer = cfg["indexer"]

    chunker = TextChunker(cfg_indexer["chunk_size"], cfg_indexer["chunk_overlap"])
    llm_client = LLMClient(cfg_core["base_url"], cfg_core["NEBIUS_API_KEY"])
    index = BuildFaissIndex(cfg_indexer["emb_dim"])

    for page_i, text in tqdm(pdf_page_text_generator(cfg_indexer["pdf_path"])):
        chunks = chunker.chunk_text(text)
        embeddings = texts2embeddings(llm_client, cfg_indexer["embedding_model"], chunks)
        index.add_embeddings(embeddings, chunks, [page_i] * len(chunks))

    index.save(cfg_indexer["processed_index_path"], cfg_indexer["processed_corpus_path"])
