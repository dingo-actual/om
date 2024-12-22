from typing import Any, Dict, List, Tuple

import orjson
from torch.utils.data import DataLoader

from src.om.data.datasets import ProportionalDataset
from src.om.data.load import get_datasets_stages
from src.om.data.tokenizer import enc


def test_run_dataloaders(config_path: str, dataloader_kwargs: Dict[str, Any]) -> Dict[str, List[int]]:
    """Test dataloaders.

    Args:
        config_path (str): path to config file
        dataloader_kwargs (Dict[str, Any]): arguments for dataloader

    Returns:
        Dict[str, List[int]]: number of tokens in train and validation dataloaders
    """
    result = {"train": [], "validation": []}
    
    dataloaders_train, dataloaders_val = get_dataloaders(config_path, dataloader_kwargs)
    
    for ix, dataloader_train in enumerate(dataloaders_train):
        result["train"].append(count_tokens_dataloader(dataloader_train))
        print(f"Train DataLoader {ix + 1}: {result['train'][-1]} tokens")
        
    for ix, dataloader_val in enumerate(dataloaders_val):
        result["validation"].append(count_tokens_dataloader(dataloader_val))
        print(f"Validation DataLoader {ix + 1}: {result['validation'][-1]} tokens")
        
    return result

def count_tokens_dataloader(dataloader: DataLoader) -> int:
    """Count tokens in dataloader.

    Args:
        dataloader (DataLoader): dataloader to count tokens in

    Returns:
        int: number of tokens in dataloader
    """
    num_tokens = 0
    for batch in dataloader:
        num_tokens += batch.size(0) * (batch.size(1) - 4)
    return num_tokens

def get_dataloaders(config_path: str, dataloader_kwargs: Dict[str, Any]) -> Tuple[List[DataLoader], List[DataLoader]]:
    """Build dataloaders.

    Args:
        config_path (str): path to config file
        dataloader_kwargs (Dict[str, Any]): arguments for dataloader

    Returns:
        Tuple[List[DataLoader], List[DataLoader]]:
        - dataloaders for training
        - dataloaders for validation
    """
    with open(config_path, "rb") as fp:
        configs = orjson.loads(fp.read())
        
    train_configs = configs["train"]
    validation_configs = configs["validation"]
    shared_config = configs["shared"]
    
    datasets_train, batch_sizes = get_datasets_split(train_configs, shared_config)
    datasets_val, _ = get_datasets_split(validation_configs, shared_config)
    
    dataloaders_train = [
        DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            **dataloader_kwargs
        )
        for dataset, batch_size in zip(datasets_train, batch_sizes)
    ]
    dataloaders_val = [
        DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            **dataloader_kwargs
        )
        for dataset, batch_size in zip(datasets_val, batch_sizes)
    ]
    
    return dataloaders_train, dataloaders_val
    
def get_datasets_split(split_config: Dict[str, Any], shared_config: Dict[str, Any]) -> Tuple[List[ProportionalDataset], List[int]]:
    """Build datasets for given split.

    Args:
        split_config (Dict[str, Any]): configuration for dataset on given split
        shared_config (Dict[str, Any]): shared configuration for all datasets

    Returns:
        Tuple[List[Dataset], List[int]]:
        - datasets for given split
        - batch sizes for given split
    """
    prefix_str = shared_config.get("prefix_str", "")
    suffix_str = shared_config.get("suffix_str", "")
    pad_str = shared_config.get("pad_str", "")
    num_pad = shared_config.get("num_pad", 0)
    
    dirs = [conf["dir"] for conf in split_config]
    match_lists = [conf["matches"] for conf in split_config]
    segment_lens = split_config[0]["segment_lens"]
    
    num_stages = len(split_config[0]["files_per_stage"])
    
    datasets_num_files = [[] for _ in range(num_stages)]
    batch_proportions = [[] for _ in range(num_stages)]
    
    for ix in range(num_stages):
        for conf in split_config:
            datasets_num_files[ix].append(conf["files_per_stage"][ix])
            batch_proportions[ix].append(conf["batch_size_per_stage"][ix])
    
    batch_sizes = [sum(batch_proportions) for batch_proportions in batch_proportions]
    
    datasets = get_datasets_stages(
        dirs=dirs,
        matches=match_lists,
        datasets_num_files=datasets_num_files,
        segment_lens=segment_lens,
        batch_sizes=batch_sizes,
        batch_proportions=batch_proportions,
        enc=enc,
        prefix_str=prefix_str,
        suffix_str=suffix_str,
        pad_str=pad_str,
        num_pad=num_pad
    )
    
    return datasets, batch_sizes
