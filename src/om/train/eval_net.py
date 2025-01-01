from accelerate import Accelerator
from evaluate import load as load_metric
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..om_llm import OmLLM


def eval_net(
    model: OmLLM, 
    # optimizer: Optimizer, 
    loss_fn: torch.nn.Module,
    dataloader_eval: DataLoader, 
    num_steps: int, 
    accelerator: Accelerator,
    batch_ix: int
) -> None:
    model = model.eval()
    # optimizer.eval()
    perpelxity = load_metric("perplexity")
    
    loss_total = 0.0
    n_tokens = 0
    
    with torch.no_grad():
        for step, batch in enumerate(dataloader_eval):
            if step == num_steps:
                break
            
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            logits = model(inputs)
            
            loss = loss_fn(logits.transpose(-1, -2), targets)
            
            accelerator.gather_for_metrics(
                (logits, targets, loss),
            )
            perpelxity.add_batch(logits, targets)
                
            batch_tokens = logits.size(0) * logits.size(1)
            loss_total += loss.cpu().detach().item() * batch_tokens
            n_tokens += batch_tokens
        
        eval_loss = loss_total / n_tokens
        
        accelerator.log(
            {
                "Loss/Validation": eval_loss,
                "Perplexity/Validation": perpelxity.compute()
            }, 
            step=batch_ix
        )
            
    model = model.train()
    # optimizer.train()
