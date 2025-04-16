#export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=bf9ae74795eb47b66c791ae2a3952ca1eacacf12
export RAY_memory_monitor_refresh_ms=0
LENGTH=200
RUN_NAME=Qwen2.5-0.5B-Instruct-sqrtlingua-${LENGTH}
MODEL=Qwen/Qwen2.5-0.5B-Instruct

N_GPUS=1
TP=1
MODEL_DIR=checkpoints/${RUN_NAME}
DATA_DIR=data/gsm8k_qwen/length${LENGTH}

BATCH_SIZE=64
ROLLOUT_BS=128
ROLLOUT_N=16



# Run the training script on the head node
echo "Starting training script on the HEAD node"
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=256 \
    data.val_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=${LENGTH} \
    actor_rollout_ref.model.path=${MODEL} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    +actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
    reward_model.enable=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.default_local_dir=$MODEL_DIR \
    trainer.default_hdfs_dir=null \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_math' \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.multisample_val=True \
    trainer.save_freq=40 \
    trainer.test_freq=20 \
    trainer.total_epochs=10 \
    trainer.num_keep_checkpoint=20 \
    trainer.resume_checkpoint=True