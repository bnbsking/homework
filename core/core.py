from typing import List, Dict

from openai import OpenAI
from tenacity import retry, wait_fixed, stop_after_attempt
import yaml


def load_yaml_to_dict(yaml_path: str) -> Dict:
    cfg = yaml.safe_load(open(yaml_path, "r"))
    return cfg


class LLMClient:
    def __init__(self, base_url: str, api_key: str, max_retries: int = 3):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
    
    @staticmethod
    def _retry_max_exceed_error(retry_state):
        raise RuntimeError("Maximum retry attempts exceeded.")

    @retry(
        wait=wait_fixed(2),
        stop=stop_after_attempt(3),
        retry_error_callback=_retry_max_exceed_error
    )
    def get_embedding(self, model: str, input: str) -> List[float]:
        response = self.client.embeddings.create(
            model=model,
            input=input
        )
        return response.data[0].embedding

    @retry(
        wait=wait_fixed(2),
        stop=stop_after_attempt(3),
        retry_error_callback=_retry_max_exceed_error
    )
    def get_chat_completion(self, model: str, messages: List[Dict]) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message.content
