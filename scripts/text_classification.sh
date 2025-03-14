CUDA_VISIBLE_DEVICES=1 python train_nll.py --dataset trec --data_split 1 --method EDA_self_teacher_v4 \
	--teacher_temperature 1 --eval_step 1 --basic_epoch 30 --da_epoch 60 --alpha_t 20 --classifier_dropout 0.1 --project_name self_teacher_v4
CUDA_VISIBLE_DEVICES=1 python train_nll.py --dataset trec --data_split 10 --method EDA_self_teacher_v4 \
	--teacher_temperature 1 --eval_step 5 --basic_epoch 30 --da_epoch 60 --alpha_t 20 --classifier_dropout 0.1 --project_name self_teacher_v4
