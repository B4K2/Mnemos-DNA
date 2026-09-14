from models.d01.attention import KDAAttention
from models.shared.gpt import GPT
from models.shared.hybrid import interleave_layer_types
from models.shared.local_attention import WindowedLocalAttention


def build_model(config: dict) -> GPT:
    layer_classes = interleave_layer_types(
        config["num_layers"], KDAAttention, WindowedLocalAttention, config.get("hybrid_ratio", 4)
    )
    return GPT(layer_classes, config)
