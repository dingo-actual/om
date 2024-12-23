import argparse
import datetime
import json
from os import makedirs
from os.path import join, exists

from accelerate import Accelerator, DDPCommunicationHookType, DistributedDataParallelKwargs
from heavyball import PaLMForeachSFAdamW, utils
import safetensors
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics.text import Perplexity

from src import *
from src.om.utils import set_om_dtypes


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
    model = set_om_dtypes(model)
    
    # Determine training stage
    for stage_ix in range(1, num_stages+1):
        final_checkpoint_dir = f"{paths_config["checkpoints"]}/stage{stage_ix}/checkpoint_FINAL"
        if not exists(final_checkpoint_dir):
            break
        
    training_config_stage = training_config[stage_ix-1]
    
    home_dir = paths_config["home"]
    checkpoint_dir = paths_config["checkpoints"]
    
    checkpoint_dir_stage = f"{checkpoint_dir}/stage{stage_ix}"
    writer_dir = f"{home_dir}/runs/stage{stage_ix}"
    
    if not exists(checkpoint_dir_stage):
        makedirs(checkpoint_dir_stage)
    if not exists(writer_dir):
        makedirs(writer_dir)
    
    if stage_ix > 1:
        checkpoint_dir_prev_stage = f"{checkpoint_dir}/stage{stage_ix-1}"
        if exists(checkpoint_dir_prev_stage):
            model.load_state_dict(safetensors.torch.load_file(checkpoint_dir_prev_stage))
    
    time_crnt = datetime.datetime.now()
    time_last = time_crnt
    
    checkpoint_dir_name = f"{checkpoint_dir_stage}/stage{stage_ix}-{time_crnt.strftime('%Y-%m-%d_%H-%M-%S')}"
    
    if not exists(checkpoint_dir_name):
        makedirs(checkpoint_dir_name)
    
    utils.set_torch()
    writer = SummaryWriter(checkpoint_dir_name)
    
    dataloader_train_kwargs = training_config_stage.pop("dataloader_train_kwargs")
    dataloader_val_kwargs = training_config_stage.pop("dataloader_val_kwargs")
    
    dataloader_train = DataLoader(train_datasets[stage_ix-1], **dataloader_train_kwargs)
    dataloader_val = DataLoader(val_datasets[stage_ix-1], **dataloader_val_kwargs)
    
    ddp_kwargs = DistributedDataParallelKwargs(comm_hook=DDPCommunicationHookType.BF16)
    accelerator = Accelerator(
        project_dir=checkpoint_dir_stage, 
        mixed_precision="bf16", 
        gradient_accumulation_steps=training_config_stage["gradient_accumulation_steps"],
        kwargs_handlers=[ddp_kwargs]
    )
    accelerator.save_state(output_dir=f"{checkpoint_dir_stage}/state_init")
    
    _ = training_config_stage.pop("gradient_accumulation_steps")
    
    opt_kwargs = training_config_stage.pop("opt_kwargs")
    log_every = training_config_stage.pop("log_every")
    eval_every = training_config_stage.pop("eval_every")
    eval_num_steps = training_config_stage.pop("eval_num_steps")
    
    # Set up optimizer
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
    
    optimizer = PaLMForeachSFAdamW(param_groups, **opt_kwargs)
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


parser = argparse.ArgumentParser()
parser.add_argument("--config_dir", type=str, default="config", help="Path to config directory")


if __name__ == "__main__":
    args = parser.parse_args()
    main(config_dir=args.config_dir)
