import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase, Qwen3Config, Qwen3ForCausalLM


def get_model(config: DictConfig, tokenizer: PreTrainedTokenizerBase) -> Qwen3ForCausalLM:
    qwen_config = Qwen3Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **config.config,
    )

    model = Qwen3ForCausalLM._from_config(
        qwen_config,
        attn_implementation=config.attn_implementation,
        dtype=getattr(torch, config.dtype),
    )
    return model
