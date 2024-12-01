# log_file_path= "/path/to/log.log"
base_model_path= "base_model/sysu/VI_sysu_BASE.pth"

# TVI-LFM training
python3 main.py --mode train --training_mode RGB_IR_Text --captioner_name Blip --llm_aug --joint_mode uni --Feat_Filter \
    --lr_txt 0.00035 --text_weight_decay 0.0005 --text_weight_decay_bias 0.0005 \
    --fusion_way add --Fix_Visual --training_weight_init $base_model_path\
    --gpu_id 0 --output_path logs/sysu  \
    --dataset sysu --loss_names id,wrt --test_modality Fusion --eval_start_epoch 0\
    --CUDA_VISIBLE_DEVICES 0 \
    # > $log_file_path 2>&1 & disown

