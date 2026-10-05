import logging
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, cast

from datasets import Dataset, load_dataset
from omegaconf import DictConfig
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


INPUT_IDS = 'input_ids'
ATTENTION_MASK = 'attention_mask'
LABELS = 'labels'


def get_tokenizer(tokenizer_name: str) -> PreTrainedTokenizerBase:
    tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(tokenizer_name))
    tokenizer.pad_token = tokenizer.eos_token
    # The pretrained tokenizer pads on the left, which shifts positions of short texts.
    tokenizer.padding_side = 'right'
    return tokenizer


def tokenize_function(
    examples: dict[str, list[Any]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> dict[str, list[list[int]]]:
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=max_length,
    )
    return {
        LABELS: [input_ids.copy() for input_ids in tokenized[INPUT_IDS]],
        INPUT_IDS: tokenized[INPUT_IDS],
        ATTENTION_MASK: tokenized[ATTENTION_MASK],
    }


def save_as_parquets(ds: Dataset, output_dir: str | Path, num_shards: int) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(num_shards):
        shard = ds.shard(num_shards=num_shards, index=index, contiguous=True)
        shard.to_parquet(output_dir / f'{index:05d}.parquet')


def load_raw_dataset(path: str, name: str, split: str, num_samples: int | None) -> Dataset:
    if num_samples is None:
        return load_dataset(path, name, split=split)
    # Streaming downloads only the first num_samples examples instead of the whole dataset.
    stream = load_dataset(path, name, split=split, streaming=True).take(num_samples)
    return Dataset.from_list(list(stream))


def prepare_dataset(
    tokenizer: PreTrainedTokenizerBase,
    path: str,
    name: str,
    split: str,
    num_samples: int | None,
    output_dir: str | Path,
    max_length: int,
    num_shards: int,
) -> None:
    dataset = load_raw_dataset(path, name, split, num_samples)
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length},
        remove_columns=dataset.column_names,
        num_proc=cpu_count(),
        desc='Tokenizing...',
    )
    save_as_parquets(tokenized_dataset, output_dir=output_dir, num_shards=num_shards)


def load_tokenized_dataset(data_dir: str | Path) -> Dataset:
    data_files = sorted(str(path) for path in Path(data_dir).glob('*.parquet'))
    dataset = load_dataset('parquet', data_files=data_files)
    return dataset['train']


def split_dataset(dataset: Dataset, validation_size: int) -> tuple[Dataset, Dataset]:
    dataset_size = len(dataset)
    train_dataset = dataset.select(range(validation_size, dataset_size))
    eval_dataset = dataset.select(range(validation_size))

    logger.info(f'Training samples: {len(train_dataset)}')
    logger.info(f'Validation samples: {len(eval_dataset)}')

    return train_dataset, eval_dataset


def get_datasets(config: DictConfig, tokenizer: PreTrainedTokenizerBase) -> tuple[Dataset, Dataset]:
    num_prepared_shards = len(list(Path(config.data_dir).glob('*.parquet')))
    if num_prepared_shards == config.num_shards:
        logger.info(f'Found tokenized dataset in {config.data_dir}')
    else:
        logger.info(f'Tokenizing dataset into {config.data_dir}')
        prepare_dataset(
            tokenizer,
            path=config.path,
            name=config.name,
            split=config.split,
            num_samples=config.num_samples,
            output_dir=config.data_dir,
            max_length=config.max_length,
            num_shards=config.num_shards,
        )

    dataset = load_tokenized_dataset(config.data_dir)
    return split_dataset(dataset, config.validation_size)
