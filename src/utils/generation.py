import torch
from omegaconf import DictConfig
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@torch.no_grad()
def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    config: DictConfig,
) -> str:
    model.eval()
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    output_ids = model.generate(  # ty:ignore[call-non-callable]
        **inputs,
        max_new_tokens=config.max_new_tokens,
        do_sample=config.do_sample,
        temperature=config.temperature,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)  # ty:ignore[invalid-return-type]
