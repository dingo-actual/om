tensorboard --logdir /home/ubuntu/om-llm/runs --host 0.0.0.0 &
accelerate launch --config_file=accelerate_config.yml pretrain_stage.py --config_dir=/home/ubuntu/om-llm/om/config &