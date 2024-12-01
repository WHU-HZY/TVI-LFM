# log_file_path= "/path/to/log.log"

# Base model training
python3 main.py --training_mode RGB_IR --mode train --gpu_id 0 \
    --dataset llcm --loss_names id,wrt --test_modality IR \
    --output_path logs/llcm/base
    # > $log_file_path 2>&1 & disown

