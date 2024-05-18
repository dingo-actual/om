import glob
import json
from os.path import join
from typing import List, Tuple

from torch.utils.data import Dataset


class TokenizedFilesDataset(Dataset):
    def __init__(self, fpaths: List[str], segment_len: int):
        """Initialize dataset.

        Args:
            fpaths (List[str]): List of paths to the jsonl files.
            segment_len (int): Length of the segments for each sample.
        """
        super(TokenizedFilesDataset, self).__init__()
        
        self.fpaths = fpaths
        self.segment_len = segment_len
        self.buffer = []
        
        self.current_file_ix = 0
        self.current_line_ix = 0
        
        self.open_current_file()
            
    def open_current_file(self):
        """Open the current file and load it into self.current_file."""
        with open(self.fpaths[self.current_file_ix], "r", encoding="utf-8") as fp:
            self.current_file = list(map(lambda st: json.loads(st.strip()), fp.readlines()))
            
    def __iter__(self):
        """Iterate over the dataset."""
        return self
    
    def __next__(self) -> List[int]:
        """Return the next sample in the dataset.
        
        Returns:
            List[int]: Current sample.
        """
        # Fill the buffer if necessary
        while len(self.buffer) < self.segment_len:
            # Read next line from current file
            if self.current_line_ix == len(self.current_file):
                # If no more lines in current file, open next file
                self.current_file_ix += 1
                self.current_line_ix = 0

                # If no more files, raise StopIteration
                if self.current_file_ix == len(self.fpaths):
                    raise StopIteration
                else:
                    self.open_current_file()
            else:
                # Add next line to buffer
                self.buffer = self.buffer + self.current_file[self.current_line_ix]
                self.current_line_ix += 1
                
        out = self.buffer[:self.segment_len]
        self.buffer = self.buffer[self.segment_len:]
                
        return out


class ProportionalDataset(Dataset):
    def __init__(self, datasets: List[Dataset], proportions: List[int]) -> None:
        """Initialize dataset.

        Args:
            datasets (List[Dataset]): Datasets to sample from.
            proportions (List[int]): List of how many samples to take from each dataset.
        """
        super().__init__()
        
        self.datasets = datasets
        self.proportions = proportions
        self.current_dataset_ix = 0
        self.current_sample_ix = 0
        
    def __iter__(self):
        """Iterate over the dataset."""
        return self
    
    def __next__(self) -> List[int]:
        """Return the next sample in the dataset.

        Returns:
            List[int]: Current sample.
        """
        # If the current sample index is equal to the proportion for the current dataset,
        # increment the current dataset index and reset the current sample index
        if self.current_sample_ix == self.proportions[self.current_dataset_ix]:
            self.current_dataset_ix += 1
            self.current_sample_ix = 0

            if self.current_dataset_ix == len(self.datasets):
                self.current_dataset_ix = 0

        # Try to sample from the current dataset
        try:
            sample = next(self.datasets[self.current_dataset_ix])
        # If the dataset is empty, set its proportion to 0 (so it'll get skipped)
        # If all datasets are empty, raise StopIteration
        except StopIteration:
            self.proportions[self.current_dataset_ix] = 0
            if all([proportion == 0 for proportion in self.proportions]):
                raise StopIteration
            else:
                return self.__next__()
        else:
            # If sampling from the current dataset was successful, increment the current sample index
            self.current_sample_ix += 1

        return sample
    
    
def get_dataset_stage(
    dirs: List[str], 
    segment_len: int, 
    max_tokens: int, 
    batch_size: int, 
    file_skip_cts: List[int], 
    dataset_proportions: List[int]
) -> Tuple[ProportionalDataset, List[int]]:
    """Create a dataset with correct proportions from the given directories.

    Args:
        dirs (List[str]): Directories for each dataset.
        segment_len (int): Segment length.
        max_tokens (int): Maximum number of tokens in the dataset.
        batch_size (int): Batch size.
        file_skip_cts (List[int]): Number of files to skip for each dataset.
        dataset_proportions (List[int]): Proportions of each dataset.

    Returns:
        ProportionalDataset: Output dataset.
        List[int]: List of updated file skip counts.
    """
    # Check that proportions are valid
    if not sum(dataset_proportions) == batch_size:
        raise ValueError("Sum of dataset proportions must equal batch size.")
    
    # Calculate fractional proportions for each dataset
    frac_proportions = [proportion / batch_size for proportion in dataset_proportions]
    
    datasets = []
    file_skip_cts_new = []
    
    # Load list of files for each dataset
    for dir, file_skip_ct, proportion in zip(dirs, file_skip_cts, frac_proportions):
        max_tokens_dataset = int(max_tokens * proportion)
        
        total_tokens = 0
        fpaths_all = sorted(glob.glob(join(dir, "*.jsonl")))
        fpaths = []
        with open(join(dir, "token_counts.txt"), "r") as fp:
            for ix, (fpath, line) in enumerate(zip(fpaths_all, fp)):
                if ix < file_skip_ct:
                    continue
                token_ct_crnt = int(line.strip())
                if total_tokens + token_ct_crnt > max_tokens_dataset:
                    break
                
                fpaths.append(fpath)
                total_tokens += token_ct_crnt
                
        datasets.append(
            TokenizedFilesDataset(
                fpaths=fpaths, 
                segment_len=segment_len
            )
        )
        file_skip_cts_new.append(ix-1)
        
    return ProportionalDataset(datasets=datasets, proportions=dataset_proportions), file_skip_cts_new


def get_datasets_stages(
    dirs: List[str], 
    segment_lens: List[int], 
    batch_sizes: List[int], 
    dataset_proportions: List[List[int]],
    stage_proportions: List[float]
) -> List[ProportionalDataset]:
    """Create datasets for each stage of optimization.

    Args:
        dirs (List[str]): Directories for each dataset.
        segment_lens (List[int]): Segment length for each stage.
        batch_size (List[int]): Batch size for each stage.
        dataset_proportions (List[List[int]]): List of dataset proportions for each stage.
        stage_proportions (List[float]): Proportions of total dataset to use for each stage.

    Returns:
        List[ProportionalDataset]: Datasets for each stage.
    """
    # Input validation
    num_stages = len(stage_proportions)
    num_datasets = len(dirs)
    
    if not len(dataset_proportions) == num_stages:
        raise ValueError("Number of dataset proportions must equal number of stages.")
    if not len(batch_sizes) == num_stages:
        raise ValueError("Number of batch sizes must equal number of stages.")
    if not len(segment_lens) == num_stages:
        raise ValueError("Number of segment lengths must equal number of stages.")
    for proportions in dataset_proportions:
        if not len(proportions) == num_datasets:
            raise ValueError("Number of dataset proportions must equal number of datasets.")
    for batch_size, proportions in zip(batch_sizes, dataset_proportions):
        if not sum(proportions) == batch_size:
            raise ValueError("Sum of dataset proportions must equal batch size.")
    
    # Get total tokens for each dataset
    dataset_total_tokens = []
    for dir in dirs:
        with open(join(dir, "total_tokens.txt"), "r") as fp:
            dataset_total_tokens.append(int(fp.read().strip()))
    total_tokens = sum(dataset_total_tokens)
    
    # Calculate number of tokens for each stage for each respective dataset
    tokens_per_stage = []
    for proportion in stage_proportions:
        tokens_per_stage.append(int(total_tokens * proportion))
        
    if sum(tokens_per_stage) != total_tokens:
        tokens_per_stage[-1] += total_tokens - sum(tokens_per_stage)
    
    # Initialize file skip numbers to zero, output to empty list
    file_skip_cts = [0] * num_datasets
    
    # Loop through stages and create each dataset
    out = []
    for (max_tokens, segment_len, batch_size, dataset_proportions_stage) in zip(tokens_per_stage, segment_lens, batch_sizes, dataset_proportions):
        dataset, file_skip_cts = get_dataset_stage(
            dirs=dirs,
            segment_len=segment_len,
            max_tokens=max_tokens,
            batch_size=batch_size,
            file_skip_cts=file_skip_cts,
            dataset_proportions=dataset_proportions_stage
        )
        out.append(dataset)
        
    return out
