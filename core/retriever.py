from abc import abstractmethod
from collections import Counter
import re
import json
from typing import List, Tuple

import numpy as np

from .core import LLMClient
from .indexer import BuildFaissIndex, Record
from .output_format import Report


class RetrieveFromFaiss:
    def __init__(
            self,
            llm_client: LLMClient,
            llm_emb_model: str,
            faiss_index: BuildFaissIndex,
        ):
        self.llm_client = llm_client
        self.llm_emb_model = llm_emb_model
        self.faiss_index = faiss_index

    def get_records(self, faiss_query: str, faiss_top_k: int) -> List[Record]:
        emb = self.llm_client.get_embedding(self.llm_emb_model, faiss_query)
        emb = np.array([emb])  # FAISS expects a 2D array
        records = self.faiss_index.search_records(emb, faiss_top_k)
        return records


class GetInfoFromRecords:
    def __init__(
            self,
            llm_client: LLMClient,
            llm_info_model: str,
        ):
        self.llm_client = llm_client
        self.llm_info_model = llm_info_model

    def run_concat(
            self,
            records: List[Record],
            llm_query: str,
        ) -> str:
        chunks = "\n\n".join([rec.chunk for rec in records])
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": llm_query.replace("{{ chunks }}", chunks)}
        ]
        answer = self.llm_client.get_chat_completion(self.llm_info_model, messages)
        return answer

    def run_separate(
            self,
            records: List[Record],
            llm_query: str,
        ) -> List[Tuple[str, int]]:
        event_page = []
        for rec in records:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": llm_query.replace("{{ chunks }}", rec.chunk)}
            ]
            event = self.llm_client.get_chat_completion(self.llm_info_model, messages)
            if event:
                event_page.append((event, rec.page))
        return event_page


class BaseInfoGetter:
    @abstractmethod
    def get_info(self, **kwargs):
        raise NotImplementedError


class InfoGetterFaiss(BaseInfoGetter):
    def __init__(
            self,
            llm_client: LLMClient,
            llm_emb_model: str,
            faiss_index: BuildFaissIndex,
            llm_info_model: str,
        ):
        self.retriever = RetrieveFromFaiss(
            llm_client,
            llm_emb_model,
            faiss_index,
        )
        self.info_getter = GetInfoFromRecords(
            llm_client,
            llm_info_model,
        )
    
    def get_info(
            self,
            faiss_query: str,
            faiss_top_k: int,
            llm_query: str,
            concat: bool = True,
        ):
        records = self.retriever.get_records(faiss_query, faiss_top_k)
        if concat:
            info_str = self.info_getter.run_concat(
                    records,
                    llm_query,
                )
            json_str = info_str.strip("```json").strip("```").replace("\n", "")
            try:
                return json.loads(json_str)
            except Exception as e:
                print("Json Decode Error", e)
                return None
        else:
            event_page = self.info_getter.run_separate(
                records,
                llm_query,
            )
            out = []
            for event, page in event_page:
                json_str = event.strip("```json").strip("```").replace("\n", "")
                try:
                    json_list = json.loads(json_str)
                    json_list = [d | {"page": int(page)} for d in json_list]
                    out.extend(json_list)
                except Exception as e:
                    print("Json Decode Error", e)
                    continue
            return out
            

class InfoGetterRe(BaseInfoGetter):
    def __init__(self, text: str):
        self.text = text
    
    def get_info(self, patterns: str):
        matches = re.findall(patterns, self.text)
        cntr = Counter(matches)
        return cntr.most_common()[0][0]
    

class InfoGetterReLLM(BaseInfoGetter):
    def __init__(
            self,
            text: str,
            llm_client: LLMClient,
            llm_info_model: str,
        ):
        self.text = text
        self.llm_client = llm_client
        self.llm_info_model = llm_info_model

    def get_info(self, patterns: str, top_k: int, query: str, key: str):
        matches = re.findall(patterns, self.text)
        cntr = Counter(matches)
        candidates = sorted(cntr.items(), key=lambda x: x[1], reverse=True)[:top_k]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query.replace("{{ candidates }}", str(candidates))}
        ]
        answer = self.llm_client.get_chat_completion(self.llm_info_model, messages)
        json_str = answer.strip("```json").strip("```").replace("\n", "")
        try:
            json_dict = json.loads(json_str)
            return json_dict[key]
        except Exception as e:
            print("Json Decode Error", e)
            return candidates[0][0]
        