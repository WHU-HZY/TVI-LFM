for trial in 1 2 3 4 5 6 7 8 9 10
do
# device=$((trial % 4))
# Baseline Tri uni Fix_Visual Filter LLM
python3 main.py --mode train \
    --training_mode RGB_IR_Text --captioner_name Blip --joint_mode uni --lr_visual 0.002\
    --fusion_way add --Fix_Visual --Feat_Filter --llm_aug --training_weight_init base_model/regdb/VI_regdb_BASE_${trail}.pth\
    --dataset regdb --loss_names id,wrt --test_modality Fusion \
    --CUDA_VISIBLE_DEVICES 0 --gpu_id 0 --trial $trial \
    --output_path logs/TVI-LFM_split${trial}
done
echo 'Done!'