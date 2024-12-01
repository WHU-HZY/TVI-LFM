# log_file_path= "/path/to/log.log"
trial=1

# Base model training
python3 main.py --training_mode RGB_IR --mode train --gpu_id 0 --lr_visual 0.002\
    --dataset regdb --loss_names id,wrt --test_modality IR \
    --output_path logs/regdb/base --trial $trial
    # > $log_file_path 2>&1 & disown

