import datetime
from typing import Any, Dict

from accelerate import Accelerator
from schedulefree import AdamWScheduleFree
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics.text import Perplexity

from .eval_net import eval_net
from ..om_llm import OmLLM
from ..data import ProportionalDataset


HOME_DIR = "/home/ubuntu/om-data"
CHECKPOINT_DIR = f"{HOME_DIR}/checkpoints"


def train_stage(
    model: OmLLM, 
    stage_num: int,
    dataset_train: ProportionalDataset, 
    dataset_val: ProportionalDataset,
    gradient_accumulation_steps: int,
    dataloader_train_kwargs: Dict[str, Any],
    dataloader_val_kwargs: Dict[str, Any],
    opt_kwargs: Dict[str, Any],
    log_every: int = 100,
    eval_every: int = 100,
    eval_num_steps: int = 100,
) -> OmLLM:
    CHECKPOINT_DIR_STAGE = f"{CHECKPOINT_DIR}/stage{stage_num}"
    WRITER_DIR = f"{HOME_DIR}/runs/stage{stage_num}"
    
    time_crnt = datetime.datetime.now()
    time_last = time_crnt
    
    writer = SummaryWriter(f"{WRITER_DIR}/{time_crnt.strftime('%Y-%m-%d_%H-%M-%S')}")
    
    accelerator = Accelerator(
        project_dir=CHECKPOINT_DIR_STAGE, 
        mixed_precision="bf16", 
        gradient_accumulation_steps=gradient_accumulation_steps
    )
    
    lr = opt_kwargs["lr"]
    adj_lr = lr * accelerator.gradient_accumulation_steps * accelerator.num_processes
    opt_kwargs["lr"] = adj_lr
    
    warmup_steps = opt_kwargs["warmup_steps"]
    adj_warmup_steps = warmup_steps * accelerator.gradient_accumulation_steps
    opt_kwargs["warmup_steps"] = adj_warmup_steps
    
    optimizer = AdamWScheduleFree(model.parameters(), **opt_kwargs)
    dataloader_train = DataLoader(dataset_train, **dataloader_train_kwargs)
    dataloader_val = DataLoader(dataset_val, **dataloader_val_kwargs)
    perplexity = Perplexity()
    loss_fn = torch.nn.CrossEntropyLoss()
    
    accelerator.register_for_checkpointing(model, optimizer, perplexity, loss_fn)
    
    model, optimizer, perplexity, loss_fn, dataloader_train, dataloader_val = accelerator.prepare(model, optimizer, perplexity, loss_fn, dataloader_train, dataloader_val)
    
    tokens_processed = 0
    
    model = model.train()
    optimizer = optimizer.train()
    
    for batch_ix, batch in enumerate(dataloader_train):
        optimizer.zero_grad()
        
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        
        tokens_processed += batch.size(0) * batch.size(1)
        
        preds, _, _ = model(inputs)
        
        with accelerator.autocast():
            loss = loss_fn(preds, targets)
        
        accelerator.backward(loss)
        
        if accelerator.sync_gradients:
            accelerator.clip_grad_value_(model.parameters(), 1.0)
        optimizer.step()
        
        time_crnt = datetime.datetime.now()
        
        if time_crnt - time_last > datetime.timedelta(minutes=60):
            time_str = time_crnt.strftime("%Y-%m-%d %H:%M:%S")
            accelerator.wait_for_everyone()
            model = model.eval()
            optimizer = optimizer.eval()
            accelerator.save_state(f"{CHECKPOINT_DIR_STAGE}/checkpoint_{time_str}.pt")
            model = model.train()
            optimizer = optimizer.train()
            time_last = time_crnt
        
        if (batch_ix + 1) % log_every == 0:
            pplx = perplexity(preds, targets)
            grad_norm = torch.sqrt(torch.sum([torch.norm(p.grad)**2 for p in model.parameters() if p.requires_grad]))
            accelerator.wait_for_everyone()
            writer.add_scalar("Tokens Processed", tokens_processed, batch_ix)
            writer.add_scalar("Loss/Train", loss.cpu().detach().item(), batch_ix)
            writer.add_scalar("Perplexity/Train", pplx.cpu().detach().item(), batch_ix)
            writer.add_scalar("Grad Norm/Train", grad_norm.cpu().detach().item(), batch_ix)
            
        if (batch_ix + 1) % eval_every == 0:
            eval_net(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                perpelxity=perplexity,
                dataloader_eval=dataloader_val,
                num_steps=eval_num_steps,
                accelerator=accelerator,
                writer=writer,
                batch_ix=batch_ix
            )

    accelerator.wait_for_everyone()
    accelerator.save_state(f"{CHECKPOINT_DIR}/checkpoint_FINAL.pt")
    
    return model