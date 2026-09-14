from models.d01.attention import KDAAttention
from models.shared.gpt import GPT


def build_model(config: dict) -> GPT:
    return GPT(KDAAttention, config)
