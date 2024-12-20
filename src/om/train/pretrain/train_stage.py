import datetime
from os import makedirs
from os.path import exists
from typing import Any, Dict

from accelerate import Accelerator, DDPCommunicationHookType, DistributedDataParallelKwargs
from heavyball import PrecondScheduleSFPaLMSOAP, utils
import safetensors
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics.text import Perplexity

from .eval_net import eval_net
from ...om_llm import OmLLM
from ...data import ProportionalDataset


def train_stage(
    model: OmLLM, 
    stage_num: int,
    paths: Dict[str, str],
    dataset_train: ProportionalDataset, 
    dataset_val: ProportionalDataset,
    gradient_accumulation_steps: int,
    dataloader_train_kwargs: Dict[str, Any],
    dataloader_val_kwargs: Dict[str, Any],
    opt_kwargs: Dict[str, Any],
    log_every: int = 100,
    eval_every: int = 100,
    eval_num_steps: int = 100,
):
    home_dir = paths["home"]
    checkpoint_dir = paths["checkpoints"]
    
    checkpoint_dir_stage = f"{checkpoint_dir}/stage{stage_num}"
    writer_dir = f"{home_dir}/runs/stage{stage_num}"
    
    makedirs(checkpoint_dir_stage)
    makedirs(writer_dir)
    
    if stage_num > 1:
        checkpoint_dir_prev_stage = f"{checkpoint_dir}/stage{stage_num-1}"
        if exists(checkpoint_dir_prev_stage):
            model.load_state_dict(safetensors.torch.load_file(checkpoint_dir_prev_stage))
    
    time_crnt = datetime.datetime.now()
    time_last = time_crnt
    
    makedirs(f"{writer_dir}/stage{stage_num}-{time_crnt.strftime('%Y-%m-%d_%H-%M-%S')}")
    writer = SummaryWriter(f"{writer_dir}/stage{stage_num}-{time_crnt.strftime('%Y-%m-%d_%H-%M-%S')}")
    
    num_pad = dataloader_train.dataset.datasets[0].num_pad
    ddp_kwargs = DistributedDataParallelKwargs(comm_hook=DDPCommunicationHookType.BF16)
    accelerator = Accelerator(
        project_dir=checkpoint_dir_stage, 
        mixed_precision="bf16", 
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs]
    )
    accelerator.save_state()
    
    utils.set_torch()
    
    lr = opt_kwargs["lr"]
    adj_lr = lr * accelerator.gradient_accumulation_steps * accelerator.num_processes
    opt_kwargs["lr"] = adj_lr
    
    warmup_steps = opt_kwargs["warmup_steps"]
    adj_warmup_steps = warmup_steps * accelerator.gradient_accumulation_steps
    opt_kwargs["warmup_steps"] = adj_warmup_steps
    
    wd_ignore_groups = ["bias", "LayerNorm"]
    wd_params = [p for n, p in model.named_parameters() if not any(nd in n for nd in wd_ignore_groups)]
    no_wd_params = [p for n, p in model.named_parameters() if any(nd in n for nd in wd_ignore_groups)]
    
    param_groups = [
        {"params": wd_params, "weight_decay": opt_kwargs["weight_decay"]},
        {"params": no_wd_params, "weight_decay": 0.0}
    ]
    _ = opt_kwargs.pop("weight_decay")
    
    optimizer = PrecondScheduleSFPaLMSOAP(param_groups, **opt_kwargs)
    dataloader_train = DataLoader(dataset_train, **dataloader_train_kwargs)
    dataloader_val = DataLoader(dataset_val, **dataloader_val_kwargs)
    perplexity = Perplexity()
    loss_fn = torch.nn.CrossEntropyLoss()
    
    accelerator.register_for_checkpointing(model, optimizer, perplexity, loss_fn)
    
    model, optimizer, perplexity, loss_fn, dataloader_train, dataloader_val = accelerator.prepare(model, optimizer, perplexity, loss_fn, dataloader_train, dataloader_val)
    
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
                grad_norm = torch.sqrt(torch.sum([torch.norm(p.grad)**2 for p in model.parameters() if p.requires_grad]))
            
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
