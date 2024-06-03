import json
from os.path import join, exists
from pathlib import Path

from src import *
from src.train.train_stage import CHECKPOINT_DIR


def main():
    # Get config filepaths
    config_path = join(Path(__file__).parent, "config")
    data_config_fpath = join(config_path, "data.json")
    model_config_fpath = join(config_path, "model.json")
    training_config_fpath = join(config_path, "training.json")
    
    # Load configs
    with open(data_config_fpath, "r") as fp:
        data_config = json.load(fp)
    with open(model_config_fpath, "r") as fp:
        model_config = json.load(fp)
    with open(training_config_fpath, "r") as fp:
        training_config = json.load(fp)
    
    num_stages = len(training_config)
    
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
    
    train_datasets = get_datasets_stages(
        dirs=[spec["dir"] for spec in data_config["train"]],
        matches=[spec["match"] for spec in data_config["train"]],
        datasets_num_files=train_num_files,
        segment_lens=data_config["train"][0]["segment_lens"],
        batch_sizes=train_batch_sizes,
        batch_proportions=train_batch_proportions
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
        matches=[spec["match"] for spec in data_config["validation"]],
        datasets_num_files=val_num_files,
        segment_lens=data_config["validation"][0]["segment_lens"],
        batch_sizes=val_batch_sizes,
        batch_proportions=val_batch_proportions
    )
    
    # Create model
    position_embedder_specs = model_config.pop("position_embedders")
    position_embedder_1 = RoPEEmbeddings(**position_embedder_specs[0])
    position_embedder_2 = RoPEEmbeddings(**position_embedder_specs[1])
    position_embedders = [position_embedder_1, position_embedder_2, position_embedder_1]
    
    model = OmLLM(position_embedders=position_embedders, **model_config)
    
    # Determine training stage
    for stage_ix in range(1, num_stages+1):
        final_checkpoint_dir = f"{CHECKPOINT_DIR}/stage{stage_ix}/checkpoint_FINAL"
        if not exists(final_checkpoint_dir):
            break
        
    training_config_stage = training_config[stage_ix-1]
    
    # Train model
    train_stage(
        model=model,
        dataset_train=train_datasets[stage_ix-1],
        dataset_val=val_datasets[stage_ix-1],
        **training_config_stage
    )


if __name__ == "__main__":
    main()
