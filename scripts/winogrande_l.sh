## Training baselines for WinoGrande-L

CUDA_VISIBLE_DEVICES=7 python qa/run_mc.py \
	--wandb_name dec_ \
	--method baseline \
	--data_dir data/winogrande/l \
	--model_type roberta \
	--model_name_or_path roberta-large \
	--task_name winogrande \
	--output_dir output/winogrande/baseline \
	--do_train \
	--do_eval \
	--evaluate_during_training \
	--per_gpu_train_batch_size 8 \
	--per_gpu_eval_batch_size 24 \
	--gradient_accumulation_steps 2 \
	--num_train_epochs 5 \
	--learning_rate 1e-5 \
	--logging_steps 1000 \
	--overwrite_output_dir \
	--max_seq_length 70 \
	--linear_decay \
	--warmup_ratio 0.06 \
	--seed 100

## Training ODDA

CUDA_VISIBLE_DEVICES=4,1 python qa/run_mc.py \
    --wandb_name teacher_ \
    --method self_CR_BC_new \
    --data_dir data/G-DAUG^C_synthetic_data/winogrande/synthetic_data/train_l \
    --train_file combo.csv \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --task_name winogrande \
    --output_dir output/winogrande/teacher_self/l \
    --do_train \
    --do_eval \
    --org_teacher \
    --teacher_path output/winogrande/self_syn_org/l/model.pt \
    --teacher_name self_syn_org \
    --teacher_method KD \
    --teacher_temperature 0.3 \
    --teacher_device 1 \
    --warmup_steps 7000 \
    --alpha_t 0.1 \
    --evaluate_during_training \
    --per_gpu_train_batch_size 16 \
    --per_gpu_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --learning_rate 5e-6 \
    --classifier_dropout 0.05 \
    --logging_steps 1000 \
    --overwrite_output_dir \
    --max_seq_length 70 \
    --linear_decay \
    --warmup_ratio 0.06 \
    --seed 100 \
    --bg_class_prior 0.0 \
    --save_model