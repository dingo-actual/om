import glob
from io import TextIOWrapper
from typing import Any, Dict, List, Tuple

import orjson
import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence
import polars as pl
from tiktoken import Encoding
import zstandard as zstd

from .datasets import ProportionalDataset, ParquetFilesDataset, CompressedJSONLFilesDataset


def get_dataset_fpaths(dir: str, match: str) -> List[str]:
    """Get the file paths for the given dataset.

    Args:
        dir (str): Dataset directory.
        match (str): File name match pattern.

    Returns:
        List[str]: File paths.
    """
    out = glob.glob(f"{dir}/**/*{match}", recursive=True)
    rs = RandomState(MT19937(SeedSequence(3487)))
    np.random.set_state(rs.get_state())
    np.random.shuffle(out)
    return out

def partition_fpaths(fpaths: List[str], num_files: List[int]) -> List[List[str]]:
    if sum(num_files) != len(fpaths):
        raise ValueError("Sum of file counts must equal number of files.")
    
    fpaths_partitioned = []
    ix_lo = 0
    for offset in num_files:
        ix_hi = min(ix_lo + offset, len(fpaths))
        fpaths_partitioned.append(fpaths[ix_lo:ix_hi])
        ix_lo = ix_hi
        
    return fpaths_partitioned

def get_dataset_stage(
    fpaths_partitioned: List[List[str]],
    segment_len: int, 
    batch_size: int, 
    batch_proportions: List[int],
    dataset_types: List[str],
    start_str: str,
    end_str: str,
    pad_str: str,
    tokenizer: Encoding,
    **kwargs
) -> ProportionalDataset:
    """Create a dataset with correct proportions from the given directories.

    Args:
        fpaths_partitioned (List[List[str]]): List of lists of file paths.
        segment_len (int): Segment length.
        batch_size (int): Batch size.
        batch_proportions (List[int]): Proportions of each dataset per batch.
        dataset_types (List[str]): List of dataset types.
        start_str (str): Start string for tokenizer.
        end_str (str): End string for tokenizer.
        pad_str (str): Padding string for tokenizer.
        tokenizer (Encoding): Tokenizer.

    Returns:
        ProportionalDataset: Output dataset.
    """
    # Input checks
    if not sum(batch_proportions) == batch_size:
        raise ValueError("Sum of batch proportions must equal batch size.")
    if not len(fpaths_partitioned) == len(batch_proportions):
        raise ValueError("Number of filepath lists and dataset proportions must be equal.")

    datasets = []
    
    for fpaths, dataset_type in zip(fpaths_partitioned, dataset_types):
        if dataset_type == "jsonl":
            ds = CompressedJSONLFilesDataset(
                fpaths=fpaths,
                segment_len=segment_len,
                start_str=start_str,
                end_str=end_str,
                pad_str=pad_str,
                tokenizer=tokenizer,
                **kwargs
            )
        elif dataset_type == "parquet":
            ds = ParquetFilesDataset(
                fpaths=fpaths,
                segment_len=segment_len,
                start_str=start_str,
                end_str=end_str,
                pad_str=pad_str,
                tokenizer=tokenizer,
                **kwargs
            )
        datasets.append(ds)
        
    return ProportionalDataset(datasets=datasets, proportions=batch_proportions)

def get_datasets_stages(
    dirs: List[str], 
    matches: List[str],
    datasets_num_files: List[List[int]],
    segment_lens: List[int], 
    batch_sizes: List[int], 
    batch_proportions: List[List[int]],
    dataset_types: List[str],
    start_str: str,
    end_str: str,
    pad_str: str,
    tokenizer: Encoding,
    **kwargs
) -> List[ProportionalDataset]:
    """Create datasets for each stage of optimization.

    Args:
        dirs (List[str]): Directories for each dataset segment.
        matches (List[str]): File name match patterns for each dataset segment.
        datasets_num_files (List[List[int]]): Number of files for each dataset segment.
        segment_lens (List[int]): Segment length for each stage.
        batch_sizes (List[int]): Batch size for each stage.
        batch_proportions (List[List[int]]): List of batch proportions for each stage.
        dataset_types (List[str]): List of dataset types.
        start_str (str): Start string for tokenizer.
        end_str (str): End string for tokenizer.
        pad_str (str): Padding string for tokenizer.
        tokenizer (Encoding): Tokenizer.

    Returns:
        List[ProportionalDataset]: Datasets for each stage.
    """
    # Input validation
    num_stages = len(batch_proportions)
    num_datasets = len(dirs)
    
    if not len(matches) == num_datasets:
        raise ValueError("Number of match expressions must equal number of datasets.")
    if not len(batch_sizes) == num_stages:
        raise ValueError("Number of batch sizes must equal number of stages.")
    if not len(datasets_num_files) == num_stages:
        raise ValueError("Number of file counts per stage must equal number of stages.")
    if not len(segment_lens) == num_stages:
        raise ValueError("Number of segment lengths must equal number of stages.")
    
    for file_cts, batch_size, proportions in zip(datasets_num_files, batch_sizes, batch_proportions):
        if not len(file_cts) == num_datasets:
            raise ValueError("Number of file counts must equal number of datasets.")
        if not sum(proportions) == batch_size:
            raise ValueError("Sum of batch proportions must equal batch size.")
        if not len(proportions) == num_datasets:
            raise ValueError("Number of dataset proportions must equal number of datasets.")

    # Get file paths for each dataset
    fpaths_all = [
        get_dataset_fpaths(dir, match) for dir, match in zip(dirs, matches)
    ]
    # Partition file paths into stages
    fpaths_partitioned = [[None for _ in range(num_datasets)] for _ in range(num_stages)]
    for ix_dataset in range(num_datasets):
        fpaths_dataset = fpaths_all[ix_dataset]
        num_files_dataset = [num_files[ix_dataset] for num_files in datasets_num_files]
        fpaths_dataset_partitioned = partition_fpaths(fpaths_dataset, num_files_dataset)
        for ix_stage, fpaths in enumerate(fpaths_dataset_partitioned):
            fpaths_partitioned[ix_stage][ix_dataset] = fpaths
    
    # Loop through stages and create each dataset
    out = []
    for (fpaths_stage, segment_len, batch_size, batch_proportions_stage) in zip(fpaths_partitioned, segment_lens, batch_sizes, batch_proportions):
        dataset = get_dataset_stage(
            fpaths_partitioned=fpaths_stage,
            segment_len=segment_len,
            batch_size=batch_size,
            batch_proportions=batch_proportions_stage,
            dataset_types=dataset_types,
            start_str=start_str,
            end_str=end_str,
            pad_str=pad_str,
            tokenizer=tokenizer,
            **kwargs
        )
        out.append(dataset)
        
    return out

def get_dataset_statistics(dir: str, match: str, **kwargs) -> Dict[str, Any]:
    """Get summary statistics for a dataset.

    Args:
        dir (str): Dataset directory.
        match (str): Match expression for dataset files.

    Returns:
        Dict[str, Any]: Summary statistics for dataset.
    """
    fpaths = get_dataset_fpaths(dir, match)
    
    num_files = len(fpaths)
    total_num_chars = 0
    
    for fpath in fpaths:
        if fpath.endswith(".parquet"):
            file = (
                pl.read_parquet(fpath)
                .get_column(kwargs["column"])
                .to_list()
            )
            for line in file:
                total_num_chars += len(line)
        elif fpath.endswith(".jsonl.zst"):
            with open(fpath, "rb") as file:
                dctx = zstd.ZstdDecompressor()
                stream_reader = dctx.stream_reader(file)
                text_wrapper = TextIOWrapper(stream_reader, encoding="utf-8")
                for line in text_wrapper:
                    total_num_chars += len(orjson.loads(line)[kwargs["field"]])
        else:
            pass
        
        num_files += 1
        
    return {
        "num_files": num_files,
        "total_num_chars": total_num_chars,
        "chars_per_file": total_num_chars / num_files
    }
