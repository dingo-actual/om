pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install xformers==0.0.28
tensorboard --logdir /home/ubuntu/om-llm/runs --host 0.0.0.0 &
accelerate launch --config_file=accelerate_config.yml pretrain_stage.py --config_dir=/home/ubuntu/om-llm/om/config