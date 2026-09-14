from models.d04.attention import HOLAAttention
from models.shared.gpt import GPT


def build_model(config: dict) -> GPT:
    return GPT(HOLAAttention, config)
