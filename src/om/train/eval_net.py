from accelerate import Accelerator
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics.text import Perplexity

from ..om_llm import OmLLM


def eval_net(
    model: OmLLM, 
    optimizer: Optimizer, 
    loss_fn: torch.nn.Module,
    perpelxity: Perplexity,
    dataloader_eval: DataLoader, 
    num_pad: int,
    num_steps: int, 
    accelerator: Accelerator,
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
            targets = batch[:, num_pad + 1:]
            
            logits, _ = model(inputs)
            
            loss = loss_fn(logits.transpose(-1, -2), targets)
            
            pplx = perpelxity(logits, targets)
            accelerator.gather_for_metrics(
                (loss, pplx),
            )
                
            batch_tokens = logits.size(0) * logits.size(1)
            loss_total += loss.cpu().detach().item() * batch_tokens
            pplx_total += pplx.cpu().detach().item() * batch_tokens
            n_tokens += batch_tokens
        
        eval_loss = loss_total / n_tokens
        eval_pplx = pplx_total / n_tokens
        
        accelerator.log(
            {
                "Loss/Validation": eval_loss,
                "Perplexity/Validation": eval_pplx
            }, 
            step=batch_ix
        )
            
    model = model.train()
    optimizer.train()
