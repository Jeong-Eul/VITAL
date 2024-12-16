export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

model_name=EHRTimeLLM
llama_layers=12

master_port=1041
num_process=2
d_model=16
d_ff=32

comment='EHRTimeLLM'

accelerate launch --config_file /home/DAHS2/.cache/huggingface/accelerate/default_config.yaml --num_processes 2 EHRTimeLLM.py \
  --dataset P19 \
  --n_heads 8 \
  --llm_model GPT2 \
  --d_model $d_model \
  --d_ff $d_ff \
  --llm_layers $llama_layers \