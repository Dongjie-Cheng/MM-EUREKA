set -x

export RAY_MASTER_PORT=6379
export RAY_DASHBOARD_PORT=8265
export NCCL_TIMEOUT=7200

OUTPUT_DIR='/absolute/path/to/output/dir'
CHAMELEON_MODEL_PATH='/path/to/chameleon-model-or-hf-id'
# NOTE: For MMK12 training, use the raw dataset fields (question, image, answer)
# and set the system prompt to mirror eval/mmk12/evaluate_mmk12.py (SYSTEM_PROMPT_7B).
SYSTEM_PROMPT_7B=$(cat <<'EOF'
Solve the question. The user asks a question, and you solves it. You first thinks about the reasoning process in the mind and then provides the user with the answer. The answer is in latex format and wrapped in $...$. The final answer must be wrapped using the \boxed{} command. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> Since $1+1=2$, so the answer is $2$. </think><answer> The answer is $\boxed{2}$ </answer>, which means assistant's output should start with <think> and end with </answer>.
EOF
)
# Chameleon image constraints (leave empty if not required, or set to match the processor limits).
IMAGE_CONSTRAINT_FLAGS=""
# Chameleon tokenizer usually supports fast tokenizers; add "--disable_fast_tokenizer" if needed.
TOKENIZER_FLAGS=""

export REWARD_LOG_PATH="${OUTPUT_DIR}/reward.log"
export WORKING_DIR=$PWD


if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

if [ "$NODE_RANK" -eq 0 ]; then
    ray start --head  --port=$RAY_MASTER_PORT --dashboard-host=0.0.0.0 --dashboard-port=$RAY_DASHBOARD_PORT --num-gpus 8
else
    sleep 30
    ray start --address="$MASTER_ADDR:$RAY_MASTER_PORT" --num-gpus 8 --block
fi

sleep 30


if [ "$NODE_RANK" -eq 0 ]; then
  RAY_ADDRESS="http://127.0.0.1:$RAY_DASHBOARD_PORT" ray job submit \
  --working-dir $WORKING_DIR \
  -- python3 -m openrlhf.cli.train_ppo_ray \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node 8 \
  --remote_rm_url examples/scripts/reward_func_qwen_instruct.py \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node 8 \
  --vllm_num_engines 8 \
  --vllm_tensor_parallel_size 1 \
  --colocate_all_models \
  --vllm_enable_sleep \
  --vllm_gpu_memory_utilization 0.3 \
  --vllm_sync_backend nccl \
  --pretrain ${CHAMELEON_MODEL_PATH} \
  --save_path ${OUTPUT_DIR} \
  --micro_train_batch_size 2 \
  --train_batch_size 128 \
  --micro_rollout_batch_size 2 \
  --rollout_batch_size 128 \
  --temperature 1.0 \
  --n_samples_per_prompt 8 \
  --lambd 1.0 \
  --gamma 1.0 \
  --max_epochs 1 \
  --num_episodes 10 \
  --prompt_max_len 3000 \
  --max_samples 100000 \
  --generate_max_len 4096 \
  --advantage_estimator group_norm \
  --zero_stage 1 \
  --bf16 \
  --actor_learning_rate 1e-6 \
  --init_kl_coef 0.0 \
  --prompt_data /path/to/training/data \
  ${IMAGE_CONSTRAINT_FLAGS} \
  ${TOKENIZER_FLAGS} \
  --apply_chat_template \
  --input_key question \
  --image_key image \
  --system_prompt "${SYSTEM_PROMPT_7B}" \
  --adam_offload \
  --flash_attn \
  --gradient_checkpointing \
  --save_steps 50 \
  --ckpt_path "${OUTPUT_DIR}/ckpt" \
  --max_ckpt_num 1000000 \
  --save_hf_ckpt \
  --freeze_prefix visual \
  --enable_accuracy_filter \
  --accuracy_lower_bound 0.1 \
  --accuracy_upper_bound 0.9 \
  --use_tensorboard "${OUTPUT_DIR}/tensorboard" \
  --load_checkpoint | tee ${OUTPUT_DIR}/training.log
fi
