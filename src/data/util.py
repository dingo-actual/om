import glob
from os.path import join
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence

from .datasets import ProportionalDataset, ParquetFilesDataset, CompressedJSONLFilesDataset, TokenizedDataset
from .tokenizer import enc


def get_dataset_fpaths(dir: str, match: str) -> List[str]:
    """Get the file paths for the given dataset.

    Args:
        dir (str): Dataset directory.
        match (str): File name match pattern.

    Returns:
        List[str]: File paths.
    """
    out = glob.glob(f"{dir}/**{match}", recursive=True)
    rs = RandomState(MT19937(SeedSequence(3487)))
    np.random.set_state(rs.get_state())
    np.random.shuffle(out)
    return out

def get_dataset_stage(
    dirs: List[str], 
    matches: List[str],
    segment_len: int, 
    max_files: List[int], 
    batch_size: int, 
    file_skip_cts: List[int], 
    batch_proportions: List[int],
    dataset_kwargs: List[Dict[str, Any]]
) -> Tuple[ProportionalDataset, List[int]]:
    """Create a dataset with correct proportions from the given directories.

    Args:
        dirs (List[str]): Directories for each dataset.
        matches (List[str]): File name match patterns for each dataset.
        segment_len (int): Segment length.
        max_files (List[int]): Maximum number of files for each component of the dataset.
        batch_size (int): Batch size.
        file_skip_cts (List[int]): Number of files to skip for each dataset.
        batch_proportions (List[int]): Proportions of each dataset per batch.
        dataset_kwargs (List[Dict[str, Any]]): Keyword arguments for each dataset.

    Returns:
        ProportionalDataset: Output dataset.
        List[int]: List of updated file skip counts.
    """
    # Input checks
    if not sum(batch_proportions) == batch_size:
        raise ValueError("Sum of batch proportions must equal batch size.")
    if not len(dirs) == len(matches) == len(max_files) == len(file_skip_cts) == len(batch_proportions):
        raise ValueError("Number of directories, matches, max files, file skip counts, and dataset proportions must be equal.")

    datasets = []
    file_skip_cts_new = []
    
    # Load list of files for each dataset
    for dir, match, max_file, file_skip_ct, kwargs in zip(dirs, matches, max_files, file_skip_cts, dataset_kwargs):
        
        fpaths = get_dataset_fpaths(dir, match)
        ix_lo = file_skip_ct
        ix_hi = min(ix_lo + max_file, len(fpaths))
        
        fpaths = fpaths[file_skip_ct:file_skip_ct+max_file]
        file_skip_cts_new.append(ix_hi)
        
        if match.endswith(".jsonl.zst"):
            ds_base = CompressedJSONLFilesDataset(
                fpaths=fpaths,
                segment_len=segment_len,
                **kwargs
            )
        elif match.endswith(".parquet"):
            ds_base = ParquetFilesDataset(
                fpaths=fpaths,
                segment_len=segment_len,
                **kwargs
            )
        ds = TokenizedDataset(ds_base, enc)
        datasets.append(ds)
        
    return ProportionalDataset(datasets=datasets, proportions=batch_proportions), file_skip_cts_new

def get_datasets_stages(
    dirs: List[str], 
    matches: List[str],
    datasets_chars_per_file: List[int],
    segment_lens: List[int], 
    batch_sizes: List[int], 
    batch_proportions: List[List[int]],
    stage_proportions: List[List[float]],
    dataset_kwargs: List[Dict[str, Any]]
) -> List[ProportionalDataset]:
    """Create datasets for each stage of optimization.

    Args:
        dirs (List[str]): Directories for each dataset segment.
        matches (List[str]): File name match patterns for each dataset segment.
        dataset_chars_per_file (List[int]): Number of characters per file for each dataset segment.
        segment_lens (List[int]): Segment length for each stage.
        batch_size (List[int]): Batch size for each stage.
        batch_proportions (List[List[int]]): List of batch proportions for each stage.
        stage_proportions (List[float]): Proportions of total dataset to use for each stage.
        dataset_kwargs (List[Dict[str, Any]]): Keyword arguments for each dataset.

    Returns:
        List[ProportionalDataset]: Datasets for each stage.
    """
    # Input validation
    num_stages = len(stage_proportions)
    num_datasets = len(dirs)
    
    if not len(batch_proportions) == num_stages:
        raise ValueError("Number of dataset proportions must equal number of stages.")
    if not len(batch_sizes) == num_stages:
        raise ValueError("Number of batch sizes must equal number of stages.")
    if not len(segment_lens) == num_stages:
        raise ValueError("Number of segment lengths must equal number of stages.")
    for proportions in batch_proportions:
        if not len(proportions) == num_datasets:
            raise ValueError("Number of batch proportions must equal number of datasets.")
    if not len(matches) == num_datasets:
        raise ValueError("Number of match expressions must equal number of datasets.")
    if not len(dataset_kwargs) == num_datasets:
        raise ValueError("Number of keyword arguments must equal number of datasets.")
    if not len(datasets_chars_per_file) == num_datasets:
        raise ValueError("Number of characters per file must equal number of datasets.")
    for batch_size, proportions in zip(batch_sizes, batch_proportions):
        if not sum(proportions) == batch_size:
            raise ValueError("Sum of batch proportions must equal batch size.")
        if not len(proportions) == num_datasets:
            raise ValueError("Number of dataset proportions must equal number of datasets.")
    
    # Calculate number of characters for each stage for each respective dataset
    files_per_dataset = [
        len(get_dataset_fpaths(dir, match))
        for dir, match in zip(dirs, matches)
    ]
    files_per_stage = [
        [int(files * proportion) for files, proportion in zip(files_per_dataset, proportions)]
        for proportions in stage_proportions
    ]
    
    files_total = [0 for _ in range(num_datasets)]
    for files_stage in files_per_stage:
        for ix, files in enumerate(files_stage):
            files_total[ix] += files
            
    for ix, files in enumerate(files_total):
        if files < files_per_dataset[ix]:
            files_per_stage[-1][ix] += files_per_dataset[ix] - files
        elif files > files_per_dataset[ix]:
            files_per_stage[-1][ix] -= files - files_per_dataset[ix]
        else:
            pass
            
    # Initialize file skip numbers to zero, output to empty list
    file_skip_cts = [0] * num_datasets
    
    # Loop through stages and create each dataset
    out = []
    for (segment_len, batch_size, batch_proportions_stage, files_stage) in zip(segment_lens, batch_sizes, batch_proportions, files_per_stage):
        dataset, file_skip_cts = get_dataset_stage(
            dirs=dirs,
            segment_len=segment_len,
            max_files=files_stage,
            batch_size=batch_size,
            file_skip_cts=file_skip_cts,
            batch_proportions=batch_proportions_stage,
            dataset_kwargs=dataset_kwargs
        )
        out.append(dataset)
        
    return out

# TODO: write function to get number of files, characters per file from a dataset