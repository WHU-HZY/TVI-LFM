# log_file_path= "/path/to/log.log"
base_model_path= "base_model/llcm/VI_llcm_BASE.pth"

# TVI-LFM training
python3 main.py --training_mode RGB_IR_Text --mode train --captioner_name Blip --joint_mode uni \
    --lr_txt 0.00035 --text_weight_decay 0.0005 --text_weight_decay_bias 0.0005 \
    --fusion_way add --Fix_Visual --Feat_Filter --llm_aug \
    --training_weight_init $base_model_path\
    --dataset llcm --loss_names id,wrt --test_modality Fusion --eval_start_epoch 0 \
    --CUDA_VISIBLE_DEVICES 0 --gpu_id 0\
    --output_path logs/llcm \
    # > $log_file_path 2>&1 & disown
