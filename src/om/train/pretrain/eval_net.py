from accelerate import Accelerator
from schedulefree import AdamWScheduleFree
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics.text import Perplexity

from ...om_llm import OmLLM


def eval_net(
    model: OmLLM, 
    optimizer: AdamWScheduleFree, 
    loss_fn: torch.nn.Module,
    perpelxity: Perplexity,
    dataloader_eval: DataLoader, 
    num_steps: int, 
    accelerator: Accelerator,
    writer: SummaryWriter,
    batch_ix: int
) -> None:
    model = model.eval()
    optimizer.eval()
    
    loss_total = 0.0
    pplx_total = 0.0
    n_tokens = 0
    
    with torch.no_grad():
        for step, batch in enumerate(dataloader_eval):
            if step == num_steps:
                break
            
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            logits = model(inputs)
            
            loss = loss_fn(logits, targets)
            pplx = perpelxity(logits, targets)
                
            batch_tokens = batch.size(0) * batch.size(1)
            loss_total += loss.cpu().detach().item() * batch_tokens
            pplx_total += pplx.cpu().detach().item() * batch_tokens
            n_tokens += batch_tokens
            
        accelerator.wait_for_everyone()
        
        eval_loss = loss_total / n_tokens
        eval_pplx = pplx_total / n_tokens
        
        writer.add_scalar("Loss/Validation", eval_loss, batch_ix)
        writer.add_scalar("Perplexity/Validation", eval_pplx, batch_ix)
            
    model = model.train()
    optimizer.train()
