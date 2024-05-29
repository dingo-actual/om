import datetime
from typing import Any, Dict

from accelerate import Accelerator
from schedulefree import AdamWScheduleFree
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics.text import Perplexity

from ..om_llm import OmLLM
from ..data import ProportionalDataset


HOME_DIR = "/home/ubuntu/om-data"
CHECKPOINT_DIR = f"{HOME_DIR}/checkpoints/stage1"


def train_stage_1(
    model: OmLLM, 
    dataset_train: ProportionalDataset, 
    batch_size: int,
    num_gpus: int,
    gradient_accumulation_steps: int,
    dataloader_train_kwargs: Dict[str, Any],
    opt_kwargs: Dict[str, Any],
    log_every: int = 100,
) -> OmLLM:
    time_crnt = datetime.datetime.now()
    time_last = time_crnt
    
    writer = SummaryWriter(f"{HOME_DIR}/runs/stage1/{time_crnt.strftime('%Y-%m-%d_%H-%M-%S')}")
    
    accelerator = Accelerator(
        project_dir=CHECKPOINT_DIR, 
        mixed_precision="bf16", 
        gradient_accumulation_steps=gradient_accumulation_steps
    )
    
    eff_batch_size = batch_size * gradient_accumulation_steps // num_gpus
    lr = opt_kwargs["lr"]
    if eff_batch_size > 0:
        adj_lr = lr * eff_batch_size / batch_size
    else:
        adj_lr = lr * gradient_accumulation_steps / num_gpus
    opt_kwargs["lr"] = adj_lr
    
    optimizer = AdamWScheduleFree(model.parameters(), **opt_kwargs)
    dataloader_train = DataLoader(dataset_train, **dataloader_train_kwargs)
    perplexity = Perplexity()
    loss_fn = torch.nn.CrossEntropyLoss()
    
    accelerator.register_for_checkpointing(model, optimizer, perplexity, loss_fn)
    
    model, optimizer, perplexity, loss_fn, dataloader_train = accelerator.prepare(model, optimizer, perplexity, loss_fn, dataloader_train)
    
    tokens_processed = 0
    
    model = model.train()
    optimizer = optimizer.train()
    
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
            
            time_crnt = datetime.datetime.now()
            
            if time_crnt - time_last > datetime.timedelta(minutes=60):
                accelerator.wait_for_everyone()
                accelerator.save_state(f"{CHECKPOINT_DIR}/checkpoint_{batch_ix:010d}.pt")
                time_last = time_crnt
            
            if (batch_ix + 1) % log_every == 0:
                accelerator.wait_for_everyone()
                writer.add_scalar("Tokens Processed", tokens_processed, batch_ix)
                writer.add_scalar("Loss/Train", loss.cpu().detach().item(), batch_ix)
                writer.add_scalar("Perplexity/Train", pplx.cpu().detach().item(), batch_ix)
                grad_norm = torch.sqrt(torch.sum([torch.norm(p.grad)**2 for p in model.parameters() if p.requires_grad]))
                writer.add_scalar("Grad Norm/Train", grad_norm.cpu().detach().item(), batch_ix)
    
    accelerator.wait_for_everyone()
    accelerator.save_state(f"{CHECKPOINT_DIR}/checkpoint_{batch_ix:010d}.pt")
    
    return model