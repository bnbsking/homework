from typing import List
import os

import faiss
import numpy as np
import pandas as pd

from .core import LLMClient


def texts2embeddings(llm_client: LLMClient, model: str, texts: List[str]) -> np.ndarray:
    embeddings = []
    for text in texts:
        embedding = llm_client.get_embedding(model, text)
        embeddings.append(embedding)
    return np.array(embeddings)  # shape = (n_texts, emb_dim)


class Record:
    chunk: str
    page: int
    score: float


class BuildFaissIndex:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.corpus = pd.DataFrame({"chunks": [], "pages": []})

    def add_embeddings(self, embeddings: np.ndarray, chunks: List[str], pages: List[int]):
        assert embeddings.shape[0] == len(chunks) == len(pages)
        self.index.add(embeddings)
        new_df = pd.DataFrame({"chunks": chunks, "pages": pages})
        self.corpus = pd.concat([self.corpus, new_df], ignore_index=True)

    def search_records(self, query_embedding: np.ndarray, top_k: int) -> List[Record]:
        assert query_embedding.shape == (1, self.index.d)
        D, I = self.index.search(query_embedding, top_k)
        records = []
        for score, i in zip(D[0], I[0]):
            record = Record()
            record.chunk = self.corpus.iloc[i]["chunks"]
            record.page = self.corpus.iloc[i]["pages"]
            record.score = score
            records.append(record)
        return records

    def save(self, index_path: str, corpus_path: str):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        self.corpus.to_csv(corpus_path, index=False)

    def load(self, index_path: str, corpus_path: str):
        self.index = faiss.read_index(index_path)
        self.corpus = pd.read_csv(corpus_path)
