# baseline

CUDA_VISIBLE_DEVICES=2 python qa/run_mc.py \
    --wandb_name dec_ \
    --data_dir data/csqa \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --task_name commonsenseqa \
    --output_dir output/csqa/baseline_single \
    --do_train --do_eval --evaluate_during_training \
    --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 8 \
    --learning_rate 1e-5 \
    --linear_decay \
    --warmup_ratio 0.06 \
    --max_seq_length 70 \
    --logging_steps 1000 --overwrite_output_dir --seed 100

# self regularization

CUDA_VISIBLE_DEVICES=4 python qa/run_mc.py \
    --wandb_name dec_ \
    --method baseline \
    --data_dir data/csqa \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --task_name commonsenseqa \
    --output_dir output/csqa/self_syn_org \
    --resume_from_checkpoint output/csqa/baseline_syn/combo.csv/roberta-large_8_epoch_1.0_decay_True_lr_5e-06_seed_100/model.pt \
    --resume_from_checkpoint_name from_baseline_ \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --per_gpu_train_batch_size 16 \
    --per_gpu_eval_batch_size 24 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 2 \
    --learning_rate 1e-5 \
    --logging_steps 1000 \
    --overwrite_output_dir \
    --max_seq_length 70 \
    --linear_decay \
    --warmup_ratio 0.06 \
    --seed 100

# ODDA

CUDA_VISIBLE_DEVICES=5,3 python qa/run_mc.py \
    --wandb_name dec_ \
    --method self_CR_BC_new \
    --data_dir data/csqa \
    --train_file train.csv \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --task_name commonsenseqa \
    --output_dir output/csqa/kd_two_iter_org \
    --do_train \
    --do_eval \
    --org_teacher \
    --teacher_path output/csqa/kd_baseline_syn_org_mix/mix_combo.csv/roberta-large_self_CR_BC_new_alpha_0.5_8_decay_True_lr5e-06_warmup_2500_10000000000.0_seed_101_p_0.0_d_p_0.05_teacher_baseline_syn_org__temp_5.0_KD/model.pt \
    --teacher_method KD \
    --teacher_temperature 1.0 \
    --teacher_device 1 \
    --alpha_t 0 \
    --evaluate_during_training \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 5 \
    --learning_rate 1e-5 \
    --classifier_dropout 0.05 \
    --logging_steps 500 \
    --overwrite_output_dir \
    --max_seq_length 70 \
    --linear_decay \
    --warmup_ratio 0.06 \
    --seed 100

CUDA_VISIBLE_DEVICES=5,3 python qa/run_mc.py \
    --wandb_name dec_ \
    --method self_CR_BC_new \
    --data_dir data/csqa \
    --train_file train.csv \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --task_name commonsenseqa \
    --output_dir output/csqa/kd_two_iter_org \
    --do_train \
    --do_eval \
    --org_teacher \
    --teacher_path output/csqa/self_syn/combo.csv/roberta-large_self_CR_BC_new_alpha_0.1_8_decay_True_lr5e-06_warmup_2000_10000000000.0_seed_100_p_0.0_d_p_0.05/model.pt \
    --teacher_method KD \
    --teacher_temperature 1.0 \
    --teacher_device 1 \
    --alpha_t 0 \
    --evaluate_during_training \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 5 \
    --learning_rate 1e-5 \
    --classifier_dropout 0.05 \
    --logging_steps 500 \
    --overwrite_output_dir \
    --max_seq_length 70 \
    --linear_decay \
    --warmup_ratio 0.06 \
    --seed 100