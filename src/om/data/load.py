import glob
from typing import List

from tiktoken import Encoding

from .datasets import ProportionalDataset, FilesDataset


def get_dataset_fpaths(dir: str, match: str) -> List[str]:
    """Get the file paths for the given dataset.

    Args:
        dir (str): Dataset directory.
        match (str): File name match pattern.

    Returns:
        List[str]: File paths.
    """
    out = glob.glob(f"{dir}/**/*{match}", recursive=True)
    out = sorted(out)

    return out

def partition_fpaths(fpaths: List[str], num_files: List[int]) -> List[List[str]]:
    if sum(num_files) > len(fpaths):
        raise ValueError("Sum of file counts must be less than or equal to the number of files.")
    
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
    enc: Encoding,
    prefix_str: str = "",
    suffix_str: str = "",
    pad_str: str = "",
    num_pad: int = 0
) -> ProportionalDataset:
    """Create a dataset with correct proportions from the given directories.

    Args:
        fpaths_partitioned (List[List[str]]): List of lists of file paths.
        segment_len (int): Segment length.
        batch_size (int): Batch size.
        batch_proportions (List[int]): Proportions of each dataset per batch.
        enc (Encoding): Tokenizer encoding.
        prefix_str (str, optional): Prefix string. Defaults to "".
        suffix_str (str, optional): Suffix string. Defaults to "".
        pad_str (str, optional): Padding string. Defaults to "".
        num_pad (int, optional): Number of padding tokens. Defaults to 0.

    Returns:
        ProportionalDataset: Output dataset.
    """
    # Input checks
    if not sum(batch_proportions) == batch_size:
        raise ValueError("Sum of batch proportions must equal batch size.")
    if not len(fpaths_partitioned) == len(batch_proportions):
        raise ValueError("Number of filepath lists and dataset proportions must be equal.")

    datasets = []
    
    for fpaths in fpaths_partitioned:
        ds = FilesDataset(
            fpaths=fpaths,
            segment_len=segment_len,
            enc=enc,
            prefix_str=prefix_str,
            suffix_str=suffix_str,
            pad_str=pad_str,
            num_pad=num_pad
        )
        datasets.append(ds)
        
    return ProportionalDataset(datasets=datasets, proportions=batch_proportions)

def get_datasets_stages(
    dirs: List[str], 
    matches: List[List[str]],
    datasets_num_files: List[List[int]],
    segment_lens: List[int], 
    batch_sizes: List[int], 
    batch_proportions: List[List[int]],
    enc: Encoding,
    prefix_str: str = "",
    suffix_str: str = "",
    pad_str: str = "",
    num_pad: int = 0
) -> List[ProportionalDataset]:
    """Create datasets for each stage of optimization.

    Args:
        dirs (List[str]): Directories for each dataset segment.
        matches (List[str]): List of file name match patterns for each dataset segment.
        datasets_num_files (List[List[int]]): Number of files for each dataset segment.
        segment_lens (List[int]): Segment length for each stage.
        batch_sizes (List[int]): Batch size for each stage.
        batch_proportions (List[List[int]]): List of batch proportions for each stage.
        enc (Encoding): Tokenizer encoding.
        prefix_str (str, optional): Prefix string. Defaults to "".
        suffix_str (str, optional): Suffix string. Defaults to "".
        pad_str (str, optional): Padding string. Defaults to "".
        num_pad (int, optional): Number of padding tokens. Defaults to 0.

    Returns:
        List[ProportionalDataset]: Datasets for each stage.
    """
    # Input validation
    num_stages = len(batch_proportions)
    num_datasets = len(dirs)
    
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
    fpaths_all = []
    for ix, dir in enumerate(dirs):
        fpaths_all.append([])
        for match in matches[ix]:
            fpaths_all[-1].append(get_dataset_fpaths(dir, match))
    
    # Partition file paths into stages
    fpaths_partitioned = [[None for _ in range(num_datasets)] for _ in range(num_stages)]
    for ix_dataset in range(num_datasets):
        fpaths_dataset = fpaths_all[ix_dataset]
        for ix_stage, fpaths in enumerate(fpaths_dataset):
            fpaths_partitioned[ix_stage][ix_dataset] = fpaths
    
    # Loop through stages and create each dataset
    out = []
    for (fpaths_stage, segment_len, batch_size, batch_proportions_stage) in zip(fpaths_partitioned, segment_lens, batch_sizes, batch_proportions):
        dataset = get_dataset_stage(
            fpaths_partitioned=fpaths_stage,
            segment_len=segment_len,
            batch_size=batch_size,
            batch_proportions=batch_proportions_stage,
            enc=enc,
            prefix_str=prefix_str,
            suffix_str=suffix_str,
            pad_str=pad_str,
            num_pad=num_pad
        )
        out.append(dataset)
        
    return out
