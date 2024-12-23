import datetime
from os import makedirs

from accelerate import Accelerator
from heavyball import PaLMForeachSFAdamW
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics.text import Perplexity

from ..eval_net import eval_net
from ...om_llm import OmLLM

def train_stage(
    model: OmLLM, 
    accelerator: Accelerator,
    optimizer: PaLMForeachSFAdamW,
    perplexity: Perplexity,
    loss_fn: torch.nn.Module,
    checkpoint_dir_stage: str,
    dataloader_train: DataLoader, 
    dataloader_val: DataLoader,
    num_pad: int,
    writer: SummaryWriter,
    time_last: datetime.datetime,
    log_every: int = 100,
    eval_every: int = 100,
    eval_num_steps: int = 100,
):
    tokens_processed = 0
    
    model = model.train()
    optimizer.train()
    
    for batch_ix, batch in enumerate(dataloader_train):
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            
            inputs = batch[:, :-1]
            targets = batch[:, num_pad + 1:]
            
            tokens_processed += batch.size(0) * (batch.size(1) - num_pad)
            
            logits, _, _ = model(inputs)
            
            with accelerator.autocast():
                loss = loss_fn(logits.transpose(-1, -2), targets)
            
            accelerator.backward(loss)
            
            if (batch_ix + 1) % log_every == 0:
                param_norm = torch.sqrt(torch.sum([torch.norm(p)**2 for p in model.parameters() if p.requires_grad])) 
                grad_norm = torch.sqrt(torch.sum([torch.norm(p.grad)**2 for p in model.parameters() if p.requires_grad and p.grad is not None]))
            
            optimizer.step()
        
        time_crnt = datetime.datetime.now()
        
        if time_crnt - time_last > datetime.timedelta(minutes=60):
            time_str = time_crnt.strftime("%Y-%m-%d %H:%M:%S")
            accelerator.wait_for_everyone()
            model = model.eval()
            optimizer.eval()
            
            checkpoint_dir_crnt = f"{checkpoint_dir_stage}/checkpoint_{time_str}"
            makedirs(checkpoint_dir_crnt)
            accelerator.save_state(checkpoint_dir_crnt)
            
            model = model.train()
            optimizer.train()
            time_last = time_crnt
        
        if (batch_ix + 1) % log_every == 0:
            pplx = perplexity(logits, targets)
            
            accelerator.wait_for_everyone()
            
            writer.add_scalar("Tokens Processed", tokens_processed, batch_ix)
            writer.add_scalar("Loss/Train", loss.cpu().detach().item(), batch_ix)
            writer.add_scalar("Perplexity/Train", pplx.cpu().detach().item(), batch_ix)
            writer.add_scalar("Parameter Norm/Train", param_norm.cpu().detach().item(), batch_ix)
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
    
    writer.add_scalar("Tokens Processed", tokens_processed, batch_ix)
    writer.add_scalar("Loss/Train", loss.cpu().detach().item(), batch_ix)
    writer.add_scalar("Perplexity/Train", pplx.cpu().detach().item(), batch_ix)
    writer.add_scalar("Grad Norm/Train", grad_norm.cpu().detach().item(), batch_ix)
    
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
    writer.flush()
    writer.close()
    
    makedirs(f"{checkpoint_dir_stage}/checkpoint_FINAL")
    accelerator.save_state(f"{checkpoint_dir_stage}/checkpoint_FINAL")
    accelerator.save_model(model, checkpoint_dir_stage)
