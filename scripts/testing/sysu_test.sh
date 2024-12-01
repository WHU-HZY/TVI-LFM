base_model_path= "base_model/sysu/VI_sysu_BASE.pth"
test_model_path= "path/to/model.pth"

python3 main.py --training_mode RGB_IR_Text --mode test --captioner_name Blip --joint_mode uni \
    --lr_txt 0.00035 --text_weight_decay 0.0005 --text_weight_decay_bias 0.0005 \
    --fusion_way add --Fix_Visual --Feat_Filter --llm_aug --test_model_path $test_model_path\
    --training_weight_init $base_model_path --output_path logs/sysu \
    --dataset llcm --loss_names id,wrt --test_modality Fusion --LOG4TEST --eval_start_epoch 0 \
    --CUDA_VISIBLE_DEVICES 0 --gpu_id 0 --CAT_EVAL --LOG4TEST