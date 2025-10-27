from dataclasses import asdict
import json

from core.core import load_yaml_to_dict, LLMClient
from core.indexer import BuildFaissIndex
from core.retriever import InfoGetterFaiss, InfoGetterRe, InfoGetterReLLM
from core.prompts import prompt_getter
from core.output_format import Report
from core.postprocess import get_age


if __name__ == "__main__":
    cfg = load_yaml_to_dict("./cfg.yaml")
    cfg_core = cfg["core"]
    cfg_indexer = cfg["indexer"]
    cfg_retriever = cfg["retriever"]

    llm_client = LLMClient(cfg_core["base_url"], cfg_core["NEBIUS_API_KEY"])
    index = BuildFaissIndex(cfg_indexer["emb_dim"])
    index.load(cfg_indexer["processed_index_path"], cfg_indexer["processed_corpus_path"])
    text = index.corpus.to_string()

    output_dict = asdict(Report())

    if "InfoGetterFaiss" in cfg_retriever["info_getters"]:
        info_getter = InfoGetterFaiss(
            llm_client,
            cfg_indexer["embedding_model"],
            index,
            cfg_retriever["info_model"],
        )
        for cfg_dict in cfg_retriever["info_getters"]["InfoGetterFaiss"]:
            info = info_getter.get_info(
                prompt_getter(cfg_dict["faiss_query"]),
                cfg_dict["faiss_top_k"],
                prompt_getter(cfg_dict["llm_query"]),
                cfg_dict["concat"]
            )
            output_dict[cfg_dict["name"]] = info
            print(info)
    
    if "InfoGetterRe" in cfg_retriever["info_getters"]:
        for cfg_dict in cfg_retriever["info_getters"]["InfoGetterRe"]:
            info_getter = InfoGetterRe(text)
            info = info_getter.get_info(
                patterns=cfg_dict["patterns"]
            )
            output_dict[cfg_dict["name"]] = info
            print(info)
    
    if "InfoGetterReLLM" in cfg_retriever["info_getters"]:
        for cfg_dict in cfg_retriever["info_getters"]["InfoGetterReLLM"]:
            info_getter = InfoGetterReLLM(text, llm_client, cfg_retriever["info_model"])
            info = info_getter.get_info(
                patterns=cfg_dict["patterns"],
                top_k=cfg_dict["top_k"],
                query=prompt_getter(cfg_dict["llm_query"]),
                key=cfg_dict["key"],
            )
            output_dict[cfg_dict["name"]] = info
            print(info)

    if "PostProcessAge" in cfg_retriever["info_getters"]:
        args = cfg_retriever["info_getters"]["PostProcessAge"]["args"]
        age = get_age(output_dict[args["birth"]])
        output_dict[args["key"]] = age

    json.dump(
        output_dict,
        open(cfg_retriever["save_path"], "w"),
        indent=4
    )
