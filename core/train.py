import torch
from tools import MultiItemAverageMeter
from torch.cuda import amp


def train(base, loaders, scaler, config, optimizer):
    base.set_train()
    meter = MultiItemAverageMeter()
    loader = loaders.get_train_loader()
    if "RN" in config.pretrain_choice:
        mode = "1/3"
    elif "VIT" in config.pretrain_choice:
        mode = None
    else: 
        ValueError(f"Pretrain model {config.pretrain_choice} choice not supported")

    # for i, (input1_0, input1_1, input2, input3, label1, label2) in enumerate(loader):
    for i, batch_dict in enumerate(loader):
        # data preparing
        # rgb_imgs0, rgb_imgs1, rgb_pids = input1_0, input1_1, label1
        # ir_imgs, ir_pids = input2, label2
        # text = input3
        # rgb_imgs0, rgb_imgs1, rgb_pids = rgb_imgs0.to(base.device), \
        #                                 rgb_imgs1.to(base.device),\
        #                                 rgb_pids.to(base.device).long()
        # ir_imgs, ir_pids = ir_imgs.to(base.device), ir_pids.to(base.device).long()
        # text = text.to(base.device).long()

        # data preparing
        batch_dict = {key: value.to(base.device) for key, value in batch_dict.items()}
        # 清空所有梯度
        optimizer.zero_grad()

        # feature and loss computing
        with amp.autocast(enabled=True):

            # get loss
            ret = base(batch_dict, mode)
            losses = [value for key, value in ret.items() if 'loss' in key]
            total_loss = sum(losses)
        
        # backward
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update meter
        acc_sign = False
        acc_value = 0
        for key, value in ret.items():
            if 'loss' in key:
                meter.update({key: value})
            if 'acc' in key:
                acc_sign = True
                acc_value = value
                # meter.update({key: value})
        meter.update({'total_loss': total_loss})
        if acc_sign:
            meter.update({'acc': acc_value})
            

    return meter.get_val(), meter.get_str()







