import torch

from .lr_scheduler import LRSchedulerWithWarmup


def build_optimizer(args, model):
    params = []

    print(f'Using {args.lr_factor} times learning rate for random init module ')
    
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.lr_visual
        weight_decay = args.visual_weight_decay
        
        if "transformer" in key:
            lr = args.lr_txt
            if "bias" in key:
                lr = args.lr_txt * args.text_bias_lr_factor
                weight_decay = args.text_weight_decay_bias
            if "cross" in key:
                lr =  args.lr_txt * args.lr_factor # default 5.0
            
        elif "visual" in key:
            lr = args.lr_visual
            if "bias" in key:
                lr = args.lr_visual * args.visual_bias_lr_factor
                weight_decay = args.visual_weight_decay_bias
            if "cross" in key:
                # use large learning rate for random initialized cross modal module
                lr =  args.lr_visual * args.lr_factor # default 5.0

        elif "classifier" in key or "mcm_head" in key or "mlm_head" in key:
                lr = args.lr_visual * args.classifier_lr_factor
        
        elif "bias" in key:
                lr = args.lr_visual * args.visual_bias_lr_factor
                weight_decay = args.visual_weight_decay_bias

        elif "cross" in key:
                # use large learning rate for random initialized cross modal module
                lr =  args.lr_visual * args.lr_factor # default 5.0

    
        
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum
        )
    elif args.optimizer == "Adam": # default
        optimizer = torch.optim.Adam(
            params,
            lr=args.lr_visual,
            weight_decay=args.visual_weight_decay,
            betas=(args.alpha, args.beta),
            eps=1e-8, # 1e-3 --> 1e-8 !!!
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-8,
        )
    else:
        NotImplementedError

    return optimizer


def build_lr_scheduler(args, optimizer):
    return LRSchedulerWithWarmup(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
        warmup_factor=args.warmup_factor,
        warmup_epochs=args.warmup_epochs,
        warmup_method=args.warmup_method,
        total_epochs=args.total_train_epoch, # 120
        mode=args.lrscheduler, # useless
        target_lr=args.target_lr, # useless
        power=args.power, # useless
    )