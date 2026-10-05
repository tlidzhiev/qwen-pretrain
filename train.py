import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import Trainer, TrainingArguments

from src.dataset import get_datasets, get_tokenizer
from src.logger import setup_comet
from src.model import get_model
from src.utils.callbacks import TimeoutCallback
from src.utils.generation import generate_text

logger = logging.getLogger(Path(__file__).name)


@hydra.main(version_base=None, config_path='src/configs', config_name='train')
def main(config: DictConfig) -> None:
    logger.info(f'Config:\n{OmegaConf.to_yaml(config, resolve=True)}')
    setup_comet(config)

    training_args = TrainingArguments(**config.training_args)
    logger.info(f'Using device: {training_args.device}')

    tokenizer = get_tokenizer(config.dataset.tokenizer_name)
    train_dataset, eval_dataset = get_datasets(config.dataset, tokenizer)

    model = get_model(config.model, tokenizer)
    logger.info(f'Model pad token id: {model.config.pad_token_id}')
    logger.info(
        f'Model:\n{model}\n'
        f'Parameters: {model.num_parameters() / 1e6:.2f}M total, '
        f'{model.num_parameters(exclude_embeddings=True) / 1e6:.2f}M non-embedding'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[TimeoutCallback(timeout_seconds=config.max_training_time_seconds)],
    )
    trainer.train()
    logger.info('Running final evaluation...')
    eval_results = trainer.evaluate()
    logger.info(f'Final evaluation results: {eval_results}')
    trainer.save_state()

    for prompt in config.generation.prompts:
        text = generate_text(trainer.model, tokenizer, prompt, config.generation)
        logger.info(f'Prompt: {prompt}\nGenerated: {text}')


if __name__ == '__main__':
    main()
