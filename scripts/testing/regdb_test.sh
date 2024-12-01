trial=1
base_model_path= "base_model/llcm/VI_regdb_BASE_${trial}.pth"
test_model_path= "path/to/model.pth"


python3 main.py --mode test \
    --training_mode RGB_IR_Text --captioner_name Blip --joint_mode uni --lr_visual 0.002 \
    --fusion_way add --Fix_Visual --Feat_Filter --llm_aug --training_weight_init $base_model_path \
    --dataset regdb --loss_names id,wrt --test_modality IR,Fusion --LOG4TEST \
    --CUDA_VISIBLE_DEVICES 0 --gpu_id 0 --trial $trial --test_model_path $test_model_path \
    --output_path logs/TVI-LFM_split1 --CAT_EVAL --LOG4TEST