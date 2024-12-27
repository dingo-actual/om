import argparse
import datetime
import json
from os import makedirs
from os.path import join, exists

from accelerate import Accelerator, DDPCommunicationHookType, DistributedDataParallelKwargs
from accelerate.utils import LoggerType
import safetensors
from schedulefree import AdamWScheduleFree
import torch
from torch.utils.data import DataLoader
from torchmetrics.text import Perplexity

from src import *
from src.om.utils import set_om_dtypes, cosine_with_warmup_mult


def main(config_dir: str):
    # Get config filepaths
    data_config_fpath = join(config_dir, "data.json")
    model_config_fpath = join(config_dir, "model.json")
    training_config_fpath = join(config_dir, "training.json")
    paths_config_fpath = join(config_dir, "paths.json")
    
    # Load configs
    with open(data_config_fpath, "r") as fp:
        data_config = json.load(fp)
    with open(model_config_fpath, "r") as fp:
        model_config = json.load(fp)
    with open(training_config_fpath, "r") as fp:
        training_config = json.load(fp)
    with open(paths_config_fpath, "r") as fp:
        paths_config = json.load(fp)
    
    num_stages = len(training_config)
    data_config_shared = data_config["shared"]
    
    # Load datasets
    train_num_files = [[] for _ in range(num_stages)]
    for spec in data_config["train"]:
        for stage_ix, num_files in enumerate(spec["files_per_stage"]):
            train_num_files[stage_ix].append(num_files)
    
    train_batch_proportions = [[] for _ in range(num_stages)]
    for spec in data_config["train"]:
        for stage_ix, batch_size in enumerate(spec["batch_size_per_stage"]):
            train_batch_proportions[stage_ix].append(batch_size)
    
    train_batch_sizes = [sum(proportions) for proportions in train_batch_proportions]
    
    if len(model_config["init_ngrams"]) == 0:
        num_pad = 0
    else:
        num_pad = max(model_config["init_ngrams"]) - 1
    
    train_datasets = get_datasets_stages(
        dirs=[spec["dir"] for spec in data_config["train"]],
        matches=[spec["matches"] for spec in data_config["train"]],
        datasets_num_files=train_num_files,
        segment_lens=data_config["train"][0]["segment_lens"],
        batch_sizes=train_batch_sizes,
        batch_proportions=train_batch_proportions,
        enc=enc,
        num_pad=num_pad,
        **data_config_shared
    )
    
    val_num_files = [[] for _ in range(num_stages)]
    for spec in data_config["validation"]:
        for stage_ix, num_files in enumerate(spec["files_per_stage"]):
            val_num_files[stage_ix].append(num_files)
    
    val_batch_proportions = [[] for _ in range(num_stages)]
    for spec in data_config["validation"]:
        for stage_ix, batch_size in enumerate(spec["batch_size_per_stage"]):
            val_batch_proportions[stage_ix].append(batch_size)
    
    val_batch_sizes = [sum(proportions) for proportions in val_batch_proportions]
    
    val_datasets = get_datasets_stages(
        dirs=[spec["dir"] for spec in data_config["validation"]],
        matches=[spec["matches"] for spec in data_config["validation"]],
        datasets_num_files=val_num_files,
        segment_lens=data_config["validation"][0]["segment_lens"],
        batch_sizes=val_batch_sizes,
        batch_proportions=val_batch_proportions,
        enc=enc,
        num_pad=num_pad,
        **data_config_shared
    )
    
    # Create model
    position_embedders = [
        RoPEEmbeddings(**position_emb_config)
        for position_emb_config in model_config["position_embedders"]
    ]
    _ = model_config.pop("position_embedders")
    
    model = OmLLM(position_embedders=position_embedders, **model_config)
    model = set_om_dtypes(model, torch.bfloat16)
    
    # Determine training stage
    for stage_ix in range(1, num_stages+1):
        final_checkpoint_dir = f"{paths_config['checkpoints']}/stage{stage_ix}/checkpoint_FINAL"
        if not exists(final_checkpoint_dir):
            break
        
    training_config_stage = training_config[stage_ix-1]
    
    # Get directories
    home_dir = paths_config["home"]
    checkpoint_dir = paths_config["checkpoints"]
    
    checkpoint_dir_stage = f"{checkpoint_dir}/stage{stage_ix}"
    writer_dir = f"{home_dir}/runs/stage{stage_ix}"
    
    # Create directories if they don't exist
    if not exists(checkpoint_dir_stage):
        makedirs(checkpoint_dir_stage)
    if not exists(writer_dir):
        makedirs(writer_dir)
    
    # If we're resuming from a previous stage, load the model
    if stage_ix > 1:
        checkpoint_dir_prev_stage = f"{checkpoint_dir}/stage{stage_ix-1}"
        if exists(checkpoint_dir_prev_stage):
            model.load_state_dict(safetensors.torch.load_file(checkpoint_dir_prev_stage))
    
    # Initialize timestamps
    time_crnt = datetime.datetime.now()
    time_last = time_crnt
    
    # Create current checkpoint directory
    checkpoint_dir_name = f"{checkpoint_dir_stage}/stage{stage_ix}-{time_crnt.strftime('%Y-%m-%d_%H-%M-%S')}"
    
    if not exists(checkpoint_dir_name):
        makedirs(checkpoint_dir_name)
    
    # Create dataloaders
    dataloader_train_kwargs = training_config_stage.pop("dataloader_train_kwargs")
    dataloader_val_kwargs = training_config_stage.pop("dataloader_val_kwargs")
    
    dataloader_train = DataLoader(train_datasets[stage_ix-1], **dataloader_train_kwargs)
    dataloader_val = DataLoader(val_datasets[stage_ix-1], **dataloader_val_kwargs)
    
    # Initialize accelerator
    ddp_kwargs = DistributedDataParallelKwargs(comm_hook=DDPCommunicationHookType.BF16)
    gradient_accumulation_steps = training_config_stage.pop("gradient_accumulation_steps")
    accelerator = Accelerator(
        project_dir=checkpoint_dir_stage, 
        mixed_precision="bf16", 
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
        log_with=LoggerType.TENSORBOARD,
    )
    accelerator.save_state(output_dir=f"{checkpoint_dir_stage}/state_init")
    
    # Extract miscellaneous training parameters
    opt_kwargs = training_config_stage.pop("opt_kwargs")
    log_every = training_config_stage.pop("log_every")
    eval_every = training_config_stage.pop("eval_every")
    eval_num_steps = training_config_stage.pop("eval_num_steps")
    grad_clip = training_config_stage.pop("gradient_clip")
    
    # Set up optimizer
    opt_kwargs["betas"] = tuple(opt_kwargs["betas"])
    
    lr = opt_kwargs["lr"]
    adj_lr = lr * accelerator.gradient_accumulation_steps * accelerator.num_processes
    opt_kwargs["lr"] = adj_lr
    
    warmup_steps = opt_kwargs.pop("warmup_steps")
    adj_warmup_steps = warmup_steps * accelerator.gradient_accumulation_steps
    total_steps = opt_kwargs.pop("total_steps")
    min_lr_mult = opt_kwargs.pop("min_lr_mult")
    
    wd_ignore_groups = ["bias", "LayerNorm"]
    wd_params = [p for n, p in model.named_parameters() if not any(nd in n for nd in wd_ignore_groups)]
    no_wd_params = [p for n, p in model.named_parameters() if any(nd in n for nd in wd_ignore_groups)]
    
    param_groups = [
        {"params": wd_params, "weight_decay": opt_kwargs["weight_decay"]},
        {"params": no_wd_params, "weight_decay": 0.0}
    ]
    _ = opt_kwargs.pop("weight_decay")
    
    # optimizer = AdamWScheduleFree(param_groups, **opt_kwargs)
    optimizer = torch.optim.AdamW(param_groups, **opt_kwargs)
    lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        optimizer=optimizer,
        lr_lambda=cosine_with_warmup_mult(
            warmup_steps=adj_warmup_steps,
            total_steps=total_steps,
            min_lr_mult=min_lr_mult
        )
    )
    
    # Initialize metrics and loss function
    perplexity = Perplexity()
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Register objects for checkpointing
    accelerator.register_for_checkpointing(
        model, 
        optimizer, 
        lr_scheduler, 
        perplexity, 
        loss_fn
    )
    
    # Prepare objects for training with Accelerate
    model, optimizer, lr_scheduler, loss_fn, dataloader_train, dataloader_val = (
        accelerator.prepare(
            model, 
            optimizer, 
            lr_scheduler, 
            loss_fn, 
            dataloader_train, 
            dataloader_val
        )
    )
    
    # Initialize training loop
    tokens_processed = 0
    
    model = model.train()
    # optimizer.train()
    
    # Main training loop
    for batch_ix, batch in enumerate(dataloader_train):
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            
            # Extract inputs and targets
            inputs = batch[:, :-1]
            targets = batch[:, num_pad + 1:]
            
            # Increment tokens processed (minus padding)
            tokens_processed += batch.size(0) * (batch.size(1) - num_pad)
            
            # Forward pass
            logits, _, _ = model(inputs)
            
            # Compute loss
            with accelerator.autocast():
                loss = loss_fn(logits.transpose(-1, -2), targets)
                
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)
            
            # Backward pass
            accelerator.backward(loss)
            
            # Log metrics
            if (batch_ix + 1) % log_every == 0 and accelerator.sync_gradients:
                param_norm = torch.sqrt(torch.sum([torch.norm(p)**2 for p in model.parameters() if p.requires_grad])) 
                grad_norm = torch.sqrt(torch.sum([torch.norm(p.grad)**2 for p in model.parameters() if p.requires_grad and p.grad is not None]))
            
            # Update parameters and perform lr step
            optimizer.step()
            lr_scheduler.step()
        
        # Update timestamp
        time_crnt = datetime.datetime.now()
        
        # If more than 1 hour has passed, save checkpoint
        if time_crnt - time_last > datetime.timedelta(minutes=60):
            time_str = time_crnt.strftime("%Y-%m-%d %H:%M:%S")
            accelerator.wait_for_everyone()
            model = model.eval()
            # optimizer.eval()
            
            # Save checkpoint
            checkpoint_dir_crnt = f"{checkpoint_dir_stage}/checkpoint_{time_str}"
            if not exists(checkpoint_dir_crnt):
                makedirs(checkpoint_dir_crnt)
            
            accelerator.save_state(checkpoint_dir_crnt)
            
            # Reset model and optimizer to training mode
            model = model.train()
            # optimizer.train()
            
            # Update timestamp
            time_last = time_crnt
        
        # Log metrics
        if (batch_ix + 1) % log_every == 0:
            accelerator.gather_for_metrics(
                (logits, targets, loss, tokens_processed, param_norm, grad_norm),
            )
            pplx = perplexity(logits, targets)
            
            accelerator.log(
                {
                    "Tokens Processed": tokens_processed,
                    "Loss/Train": loss.cpu().detach().item(),
                    "Perplexity/Train": pplx.cpu().detach().item(),
                    "Parameter Norm/Train": param_norm.cpu().detach().item(),
                    "Grad Norm/Train": grad_norm.cpu().detach().item()
                }, 
                step=batch_ix
            )
        
        # Evaluate model on validation set
        if (batch_ix + 1) % eval_every == 0:
            accelerator.wait_for_everyone()
            eval_net(
                model=model,
                # optimizer=optimizer,
                loss_fn=loss_fn,
                perpelxity=perplexity,
                dataloader_eval=dataloader_val,
                num_steps=eval_num_steps,
                accelerator=accelerator,
                batch_ix=batch_ix
            )

    # Final evaluation and metrics at end of training
    accelerator.gather_for_metrics(
        (logits, targets, loss, tokens_processed, param_norm, grad_norm),
    )
    pplx = perplexity(logits, targets)
    
    param_norm = torch.sqrt(torch.sum([torch.norm(p)**2 for p in model.parameters() if p.requires_grad])) 
    grad_norm = torch.sqrt(torch.sum([torch.norm(p.grad)**2 for p in model.parameters() if p.requires_grad and p.grad is not None]))
    
    accelerator.log(
        {
            "Tokens Processed": tokens_processed,
            "Loss/Train": loss.cpu().detach().item(),
            "Perplexity/Train": pplx.cpu().detach().item(),
            "Parameter Norm/Train": param_norm.cpu().detach().item(),
            "Grad Norm/Train": grad_norm.cpu().detach().item()
        }, 
        step=batch_ix
    )
    
    accelerator.wait_for_everyone()
    eval_net(
        model=model,
        # optimizer=optimizer,
        loss_fn=loss_fn,
        perpelxity=perplexity,
        dataloader_eval=dataloader_val,
        num_steps=eval_num_steps,
        accelerator=accelerator,
        batch_ix=batch_ix
    )
    
    # Save final checkpoint
    checkpoint_dir_final = f"{checkpoint_dir_stage}/checkpoint_FINAL"
    if not exists(checkpoint_dir_final):
        makedirs(checkpoint_dir_final)
        
    accelerator.save_state(f"{checkpoint_dir_stage}/checkpoint_FINAL")
    accelerator.save_model(model, checkpoint_dir_stage)
    
    accelerator.end_training()


# Define command line parser
parser = argparse.ArgumentParser()
parser.add_argument("--config_dir", type=str, default="config", help="Path to config directory")


# ENTRY POINT: parse arguments and run main function
if __name__ == "__main__":
    args = parser.parse_args()
    main(config_dir=args.config_dir)
