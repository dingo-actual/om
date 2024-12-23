mkdir -p /home/ubuntu/om-llm/om/runs
tensorboard --logdir /home/ubuntu/om-llm/om/runs &
accelerate launch --config-path=accelerate_config.yml pretrain_stage.py --config_dir=/home/ubuntu/om-llm/om/config