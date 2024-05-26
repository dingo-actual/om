import glob
from io import TextIOWrapper
from typing import Any, Dict, List, Tuple

import orjson
import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence
import polars as pl
import zstandard as zstd

from .datasets import ProportionalDataset, ParquetFilesDataset, CompressedJSONLFilesDataset
from .tokenizer import enc


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

def get_dataset_stage(
    dirs: List[str], 
    matches: List[str],
    segment_len: int, 
    num_files: List[int], 
    batch_size: int, 
    file_skip_cts: List[int], 
    batch_proportions: List[int],
    **kwargs
) -> Tuple[ProportionalDataset, List[int]]:
    """Create a dataset with correct proportions from the given directories.

    Args:
        dirs (List[str]): Directories for each dataset.
        matches (List[str]): File name match patterns for each dataset.
        segment_len (int): Segment length.
        num_files (List[int]): Number of files for each component of the dataset.
        batch_size (int): Batch size.
        file_skip_cts (List[int]): Number of files to skip for each dataset.
        batch_proportions (List[int]): Proportions of each dataset per batch.

    Returns:
        ProportionalDataset: Output dataset.
        List[int]: List of updated file skip counts.
    """
    # Input checks
    if not sum(batch_proportions) == batch_size:
        raise ValueError("Sum of batch proportions must equal batch size.")
    if not len(dirs) == len(matches) == len(num_files) == len(file_skip_cts) == len(batch_proportions):
        raise ValueError("Number of directories, matches, max files, file skip counts, and dataset proportions must be equal.")

    datasets = []
    file_skip_cts_new = []
    
    # Load list of files for each dataset
    for dir, match, max_file, file_skip_ct in zip(dirs, matches, num_files, file_skip_cts):
        
        fpaths = get_dataset_fpaths(dir, match)
        ix_lo = file_skip_ct
        ix_hi = min(ix_lo + max_file, len(fpaths))
        
        fpaths = fpaths[file_skip_ct:file_skip_ct+max_file]
        file_skip_cts_new.append(ix_hi)
        
        if match.endswith(".jsonl.zst"):
            ds = CompressedJSONLFilesDataset(
                fpaths=fpaths,
                segment_len=segment_len,
                **kwargs
            )
        elif match.endswith(".parquet"):
            ds = ParquetFilesDataset(
                fpaths=fpaths,
                segment_len=segment_len,
                **kwargs
            )
        datasets.append(ds)
        
    return ProportionalDataset(datasets=datasets, proportions=batch_proportions), file_skip_cts_new

def get_datasets_stages(
    dirs: List[str], 
    matches: List[str],
    datasets_num_files: List[List[int]],
    segment_lens: List[int], 
    batch_sizes: List[int], 
    batch_proportions: List[List[int]],
    **kwargs
) -> List[ProportionalDataset]:
    """Create datasets for each stage of optimization.

    Args:
        dirs (List[str]): Directories for each dataset segment.
        matches (List[str]): File name match patterns for each dataset segment.
        datasets_num_files (List[int]): Number of files for each dataset segment.
        segment_lens (List[int]): Segment length for each stage.
        batch_size (List[int]): Batch size for each stage.
        batch_proportions (List[List[int]]): List of batch proportions for each stage.

    Returns:
        List[ProportionalDataset]: Datasets for each stage.
    """
    # Input validation
    num_stages = len(batch_proportions)
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
    if not len(datasets_num_files) == num_datasets:
        raise ValueError("Number of file counts must equal number of datasets.")
    for batch_size, proportions in zip(batch_sizes, batch_proportions):
        if not sum(proportions) == batch_size:
            raise ValueError("Sum of batch proportions must equal batch size.")
        if not len(proportions) == num_datasets:
            raise ValueError("Number of dataset proportions must equal number of datasets.")

    # Initialize file skip numbers to zero, output to empty list
    file_skip_cts = [0] * num_datasets
    
    # Loop through stages and create each dataset
    out = []
    for (segment_len, match, batch_size, batch_proportions_stage, files_stage) in zip(segment_lens, matches, batch_sizes, batch_proportions, datasets_num_files):
        dataset, file_skip_cts = get_dataset_stage(
            dirs=dirs,
            matches=match,
            segment_len=segment_len,
            num_files=files_stage,
            batch_size=batch_size,
            file_skip_cts=file_skip_cts,
            batch_proportions=batch_proportions_stage,
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