from typing import List

import orjson
from tiktoken import Encoding
import torch
from torch.utils.data import IterableDataset


class FilesDataset(IterableDataset):
    def __init__(
        self, 
        fpaths: List[str], 
        segment_len: int,
        enc: Encoding,
        prefix_str: str = "",
        suffix_str: str = "",
        pad_str: str = "",
        num_pad: int = 0
    ):
        """Initialize dataset.

        Args:
            fpaths (List[str]): List of paths to the files.
            segment_len (int): Length of the segments for each sub-sample.
        """
        super(FilesDataset, self).__init__()
        
        self.fpaths = fpaths
        self.segment_len = segment_len
        self.enc = enc
        self.prefix_str = prefix_str
        self.suffix_str = suffix_str
        self.pad_str = pad_str
        
        self.num_pad = num_pad
        if num_pad <= 0:
            self.pad_token = -1
        elif pad_str == "":
            self.num_pad = 0
            self.pad_token = -1
        else:
            self.pad_token = self.enc.encode_single_token(pad_str)

        self.buffer = []
        
        self.current_file_ix = 0
        self.current_obs_ix = 0
        
        self.open_current_file()
        
        self.current_file_ix += 1

    def open_current_file(self):
        """Open the next file."""
        with open(self.fpaths[self.current_file_ix], "r") as fp:
            current_file = fp.read()
        
        current_file = self.parse(current_file)
        current_file = [
            self.prefix_str + line.strip() + self.suffix_str 
            for line in current_file
        ]
        self.current_file = self.enc.encode_batch(current_file)
            
    def __iter__(self):
        """Iterate over the dataset."""
        return self
    
    def __next__(self) -> torch.Tensor:
        """Return the next sample in the dataset.
        
        Returns:
            str: Current sample.
        """
        # Fill the buffer if necessary
        while len(self.buffer) < self.segment_len - self.num_pad:
            # Read next line from current file
            if self.current_obs_ix == len(self.current_file):
                # If no more lines in current file, open next file
                self.current_file_ix += 1
                self.current_obs_ix = 0

                # If no more files, raise StopIteration
                if self.current_file_ix == len(self.fpaths):
                    raise StopIteration
                else:
                    self.open_current_file()
            else:
                # Add next line to buffer
                self.buffer.extend(self.current_file[self.current_obs_ix])
                self.current_obs_ix += 1
        
        padding = [self.pad_token for _ in range(self.num_pad)]
        out = padding + self.buffer[:self.segment_len - self.num_pad]
        self.buffer = self.buffer[self.segment_len - self.num_pad:]
                
        return torch.tensor(out)
    
    def parse(self, text: str) -> List[str]:
        return orjson.loads(text.strip())

class ProportionalDataset(IterableDataset):
    def __init__(
        self, 
        datasets: List[IterableDataset], 
        proportions: List[int]
    ) -> None:
        """Initialize dataset.

        Args:
            datasets (List[IterableDataset]): Datasets to sample from.
            proportions (List[int]): List of how many samples to take from each dataset.
        """
        super().__init__()
        
        self.datasets = datasets
        self.proportions = proportions
        self.current_dataset_ix = 0
        self.current_sample_ix = 0
        
        self.skip = [False if p > 0 else True for p in self.proportions]
        
    def __iter__(self):
        """Iterate over the dataset."""
        return self
    
    def __next__(self) -> torch.Tensor:
        """Return the next sample in the dataset.

        Returns:
            torch.Tensor: Current sample.
        """
        # If the current sample index is equal to the proportion for the current dataset,
        # increment the current dataset index and reset the current sample index
        if self.current_sample_ix >= self.proportions[self.current_dataset_ix] or self.skip[self.current_dataset_ix]:
            self.current_dataset_ix = (self.current_dataset_ix + 1) % len(self.datasets)
            self.current_sample_ix = 0

        # Try to sample from the current dataset
        try:
            sample = next(self.datasets[self.current_dataset_ix])
        # If the dataset is empty, set its proportion to 0 (so it'll get skipped)
        # If all datasets are empty, raise StopIteration
        except StopIteration:
            self.skip[self.current_dataset_ix] = True
            if all(self.skip):
                raise StopIteration
            else:
                return self.__next__()
        else:
            # If sampling from the current dataset was successful, increment the current sample index
            self.current_sample_ix += 1

        return sample
