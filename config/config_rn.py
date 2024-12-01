import argparse
import ast


def get_args():
    parser = argparse.ArgumentParser()
    # Debug
    parser.add_argument('--DEBUG', default=False, action='store_true')
    parser.add_argument('--DEBUG_DIR', default='/data0/hzy_log/WORK_2024_LOG/Debug_logs', type=str, help='debug dir')

    # Device and DataParallel setting 
    parser.add_argument('--cuda', type=str, default='cuda')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default="0,1,2,3", help="0,1,2,3")
    parser.add_argument('--gpu_id', type=str, default='0',help='using single gpu')
    parser.add_argument('--DataParallel', default=False, action='store_true', help='whether to use DataParallel')

    # training and testing config
    parser.add_argument('--LOG4TEST', default=False, action='store_true') # config file path
    parser.add_argument('--config_select', type=str, default='default', help='config file path') # config file path 
    parser.add_argument('--mode', type=str, default='train', help='train, test')
    parser.add_argument('--test_mode', default='all', type=str, help='all or indoor')
    parser.add_argument('--gall_mode', default='single', type=str, help='single or multi')
    parser.add_argument('--regdb_test_mode', default='t-v', type=str, help='')
    parser.add_argument('--resume_test_model', type=int, default=-1, help='-1 for no resuming')
    parser.add_argument('--test_model_type', default='Fusion', help='the type of mode for testing["IR","Fusion","Text"]',type=str)
    parser.add_argument('--test_modality', default='Fusion', help='testing retrieval mode ["IR","Fusion","Text"]',type=str)
    parser.add_argument('--training_mode', default='RGB_IR_Text', type=str, help='RGB_Text, RGB_IR, RGB_IR_Text')
    parser.add_argument('--joint_mode', default='ir_crossfusion', type=str, help='[ir_crossfusion, ir_selffusion, rgb_selffusion, uni, dual_text]')



    # LLM augmentation setting 
    parser.add_argument('--llm_aug', default=False, action='store_true', help='whether use llm augmentation in training or not')
    parser.add_argument('--llm_aug_prob', default=0.5, type=float, help='prob for applying llm rephrase')
    # parser.add_argument('--text_guided', default=False, action='store_true', help='whether use text guided and related model to train')
    parser.add_argument('--captioner_name', default='Blip', type=str, help='Use which Captioner type to do the augmentation ["GIT","Blip"]')

    # dataset setting 
    parser.add_argument('--Feat_Filter', default=False, action='store_true')
    parser.add_argument('--training_weight_init', default=None, type=str, help='weight path initialization for training')
    parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu or llcm]')
    parser.add_argument('--sysu_data_path', type=str, default='/data0/hzy_data/SYSU-MM01/')
    parser.add_argument('--regdb_data_path', type=str, default='/data0/hzy_data/RegDB/')
    parser.add_argument('--llcm_data_path', type=str, default='/data0/hzy_data/LLCM/')
    parser.add_argument('--pretrain_path', type=str, default='default', help='pretrained model path')
    parser.add_argument('--trial', default=1, type=int, help='trial (only for RegDB dataset)')
    parser.add_argument('--eval_num_regdb', default=1, type=int) # 1 or 10 for regdb
    parser.add_argument('--batch-size', default=32, type=int, metavar='B', help='training batch size')
    parser.add_argument('--img_w', default=144, type=int, metavar='imgw', help='img width')
    parser.add_argument('--img_h', default=288, type=int, metavar='imgh', help='img height')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--pid_num', type=int, default=395)
    parser.add_argument('--num_pos', default=4, type=int,help='num of pos per identity in each modality')
    parser.add_argument('--num_workers', default=8, type=int,help='num of pos per identity in each modality')

 
    ######################## eval and log config during training ########################
    parser.add_argument('--test_model_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default='/data0/hzy_log/WORK_2024_LOG/logs/',
                        help='path to save related informations')
    parser.add_argument('--max_save_model_num', type=int, default=1, help='0 for max num is infinit')
    parser.add_argument('--resume_train_epoch', type=int, default=-1, help='-1 for no resuming')
    parser.add_argument('--auto_resume_training_from_lastest_step', action="store_true", default=False)
    parser.add_argument('--eval_epoch', type=int, default=2)
    parser.add_argument('--eval_start_epoch', type=int, default=80)
    parser.add_argument('--checkpoint_epoch', type=int, default=10)
    parser.add_argument('--CAT_EVAL', default=False, action='store_true')

 
    ######################## model general settings ########################
    parser.add_argument('--uni_BN', default=False, action='store_true')
    parser.add_argument('--Fix_Visual', default=False, action='store_true')
    parser.add_argument('--Return_B4_BN', default=False, action='store_true')
    parser.add_argument('--prj_output_dim', type=int, default=2048)
    parser.add_argument("--pooling", default='GEM', type=str, help='["attnpool","GEM"]')
    parser.add_argument("--pretrain_choice", default='RN50_ORI',help='ViT-B/16,RN50,RN50_ORI') # whether use pretrained model
    parser.add_argument("--temperature", type=float, default=0.07, help="initial temperature value, if 0, don't use temperature")

    ######################## cross transfomer setting ########################
    parser.add_argument("--cmt_depth", type=int, default=1, help="cross modal transformer self attn layers")
    parser.add_argument("--lr_factor", type=float, default=5.0, help="lr factor for random init self implement module")

    ######################## loss settings ########################
    # parser.add_argument("--uni_train", default=False, action='store_true', help="whether use uni_modal training")
    parser.add_argument("--loss_names", default='wrt,id', help="which loss to use ['wrt', 'id', 'uni_reid', 'Fusion_Regular']")
    parser.add_argument("--cmm_loss_weight", type=float, default=1.0, help="cross modal matching loss (tcmpm, cmpm, infonce...) weight")
    parser.add_argument("--id_loss_weight", type=float, default=1.0, help="id loss weight")
    parser.add_argument("--wrt_loss_weight", type=float, default=1.0, help="itc loss weight")
    
    ######################## vison trainsformer settings ########################
    parser.add_argument("--img_size", type=tuple, default=(288, 144))
    parser.add_argument("--stride_size", type=int, default=16)

    ######################## text transformer settings ########################
    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408)

    ######################## solver ########################
    parser.add_argument("--learnable_loss_weight", default=False)
    parser.add_argument("--label_mix", default=False, action='store_true', help="whether mix pid and imagid label")
    parser.add_argument("--optimizer", type=str, default="Adam", help="[SGD, Adam, Adamw]")

    parser.add_argument("--lr_txt", type=float, default=1e-5) # 如果使用文字模态的transformer，需要设置一个较小的学习率
    parser.add_argument("--text_weight_decay", type=float, default=4e-5) 
    parser.add_argument("--text_bias_lr_factor", type=float, default=2.) 
    parser.add_argument("--text_weight_decay_bias", type=float, default=0.) # for text 4e-5
    parser.add_argument("--lr_visual", type=float, default=0.00035) # 两个gpu就乘以二 7e-4 # RegDB 0.001
    parser.add_argument("--visual_weight_decay", type=float, default=5e-4)
    parser.add_argument("--visual_bias_lr_factor", type=float, default=1.) 
    parser.add_argument("--visual_weight_decay_bias", type=float, default=5e-4)

    parser.add_argument("--classifier_lr_factor", type=float, default=2.)
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum for SGD")
    parser.add_argument("--alpha", type=float, default=0.9, help='for adam and adamW')
    parser.add_argument("--beta", type=float, default=0.999, help='for adam and adamW')
    
    ######################## scheduler ########################
    parser.add_argument('--total_train_epoch', type=int, default=120) # 100
    parser.add_argument("--milestones", type=int, nargs='+', default=(40, 60, 100)) # (40,60)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_factor", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--warmup_method", type=str, default="linear")
    parser.add_argument("--lrscheduler", type=str, default="step")
    parser.add_argument("--target_lr", type=float, default=0.0)
    parser.add_argument("--power", type=float, default=0.9) # 1.0

    ######################## multi-modality model settings ########################
    parser.add_argument("--fusion_way", default='add', help="[add, weight_add, cross_attention, norm_add]") # whether use text and sketch fusion method
    parser.add_argument("--only_sketch", default=False, action='store_true', help="whether training with only sketch")
    parser.add_argument("--only_text", default=False, action='store_true', help="whether training with only text")
    parser.add_argument("--pa", type=float, default=0.1, help="parameter add for fusion")
    parser.add_argument("--only_fusion_loss", default=False, action='store_true', help="whether training with only text")
    parser.add_argument("--four_fusion_loss", default=False, action='store_true', help="whether training with only text")
    parser.add_argument("--focal_three_fusion_loss", default=False, action='store_true', help="whether training with only text")
    parser.add_argument("--focal_three_fusion_loss2", default=False, action='store_true', help="whether training with only text")
    parser.add_argument("--focal_three_fusion_loss3", default=False, action='store_true', help="sketch label kl")
    parser.add_argument("--focal_three_fusion_loss4", default=False, action='store_true', help=" text label kl")
    parser.add_argument("--focal_three_fusion_loss5", default=False, action='store_true', help=" text label two kl")
    parser.add_argument("--focal_three_fusion_loss6", default=False, action='store_true', help=" text label two kl")
    parser.add_argument("--focalthree_fusion_loss", default=False, action='store_true', help="whether training with only text")
    parser.add_argument("--focalthree_four_fusion_loss", default=False, action='store_true', help="whether training with only text")
    parser.add_argument("--al", type=float, default=1.0, help="parameter add for fusion")
    parser.add_argument("--ga", type=float, default=2.0, help="parameter add for fusion")
    parser.add_argument("--klp", type=float, default=1.0, help="parameter add for fusion")
    
    config = parser.parse_args()

    return config