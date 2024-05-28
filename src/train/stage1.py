import datetime
from typing import Any, Dict

from accelerate import Accelerator
from schedulefree import AdamWScheduleFree
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics.text import Perplexity

from ..om_llm import OmLLM
from ..data import ProportionalDataset, CompressedJSONLFilesDataset


HOME_DIR = "/home/ubuntu/om-data"
CHECKPOINT_DIR = f"{HOME_DIR}/checkpoints/stage1"


def train_stage_1(
    model: OmLLM, 
    dataset_train: ProportionalDataset, 
    dataloader_train_kwargs: Dict[str, Any],
    dataset_val: CompressedJSONLFilesDataset,
    dataloader_val_kwargs: Dict[str, Any],
    opt_kwargs: Dict[str, Any],
    checkpoint_every: int,
) -> OmLLM:
    writer = SummaryWriter(f"{HOME_DIR}/runs/stage1/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    
    accelerator = Accelerator(project_dir=CHECKPOINT_DIR, mixed_precision="fp16")
    
    optimizer = AdamWScheduleFree(model.parameters(), **opt_kwargs)
    
    dataloader_train = DataLoader(dataset_train, **dataloader_train_kwargs)
    dataloader_val = DataLoader(dataset_val, **dataloader_val_kwargs)
    
    perplexity = Perplexity()
    loss_fn = torch.nn.CrossEntropyLoss()
    
    model, optimizer, perplexity, loss_fn, dataloader_train, dataloader_val = accelerator.prepare(model, optimizer, perplexity, loss_fn, dataloader_train, dataloader_val)
    
    tokens_processed = 0
    
    model = model.train()
    optimizer = optimizer.train()
    
    # TODO: add hooks for tensorboard
    for batch_ix, batch in enumerate(dataloader_train):
        with accelerator.autocast():
            optimizer.zero_grad()
            
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            tokens_processed += batch.size(0) * batch.size(1)
            
            preds, _, _ = model(inputs)
            
            loss = loss_fn(preds, targets)
            pplx = perplexity(preds, targets)
            
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()
            
            # TODO: add checkpoint logic
            if (batch_ix + 1) % checkpoint_every == 0:
                pass