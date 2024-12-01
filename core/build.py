import os
from tools import os_walk
import network.clip_model.objectives as objectives
from network.clip_model.clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import TripletLoss_WRT,kl_align_loss,TripletLoss_WRT_local,L_i2t,L_t2i

    
class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('InstanceNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class Classifier(nn.Module):
    def __init__(self, pid_num, dim=512, Return_B4_BN=False, uni_BN=False, joint_mode='uni',modal='RGB,IR,Text,Fusion'):
        super(Classifier, self, ).__init__()
        self.pid_num = pid_num
        # self.GAP = GeneralizedMeanPoolingP()
        self.Return_B4_BN = Return_B4_BN
        self.modal = modal
        self.uni_BN = uni_BN
        self.joint_mode = joint_mode
        if uni_BN:
            assert joint_mode == 'uni'
            if joint_mode == 'uni':
                self.BN_RGB = nn.BatchNorm1d(dim)
                self.BN_RGB.apply(weights_init_kaiming)
                self.BN_IR = nn.BatchNorm1d(dim)
                self.BN_IR.apply(weights_init_kaiming)
                self.BN_Fusion = nn.BatchNorm1d(dim)
                self.BN_Fusion.apply(weights_init_kaiming)
                self.BN_Text = nn.BatchNorm1d(dim)
                self.BN_Text.apply(weights_init_kaiming)
        else:
            self.BN = nn.BatchNorm1d(dim)
            self.BN.apply(weights_init_kaiming)

        self.classifier = nn.Linear(dim, self.pid_num, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.l2_norm = Normalize(2)

    def forward(self, features, mode="RGB"): # IR, Fusion, Text, RGB
        # features = self.GAP(features_map)
        if self.uni_BN:
            if self.training:
                len_feat = len(features)
                b = len_feat // 5
                rgb_features = self.BN_RGB(features[:2*b].squeeze())
                ir_features = self.BN_IR(features[2*b:3*b].squeeze())
                fusion_features = self.BN_Fusion(features[3*b:4*b].squeeze())
                text_features = self.BN_Text(features[4*b:5*b].squeeze())
                bn_features = torch.cat((rgb_features, ir_features, fusion_features, text_features),dim=0)
                
            else:
                if mode == 'RGB':
                    bn_features = self.BN_RGB(features.squeeze())
                elif mode == 'IR':
                    bn_features = self.BN_IR(features.squeeze())
                elif mode == 'Fusion':
                    bn_features = self.BN_Fusion(features.squeeze())
                elif mode == 'Text':
                    bn_features = self.BN_Text(features.squeeze())
                else:
                    raise ValueError("mode must be in ['IR', 'Fusion', 'Text', 'RGB']")
        else:
            bn_features = self.BN(features.squeeze())

        cls_score = self.classifier(bn_features)

        if self.training:
            return features, cls_score
        else:
            # if self.Return_B4_BN:
            #     return features
            return self.l2_norm(bn_features)


class FM_cat(nn.Module):
    def __init__(self,in_channels):
        super(FM_cat, self).__init__()

        self.W = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels,
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_channels)
        )
        nn.init.normal_(self.W[1].weight.data, 1.0, 0.01)
        nn.init.zeros_(self.W[1].bias.data)


        # self.bottleneck = nn.BatchNorm1d(in_channels)
        # self.bottleneck.bias.requires_grad_(False)  # no shift

        # nn.init.normal_(self.bottleneck.weight.data, 1.0, 0.01)
        # nn.init.zeros_(self.bottleneck.bias.data)

    def forward(self,f):

        f = f.view(f.size(0),f.size(1),1,1)
        f = self.W(f)
        f = f.view(f.size(0),-1)
        # f = self.bottleneck(f+feat)

        return f

class CLIP2ReID(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.max_save_model_num = args.max_save_model_num
        self.output_path = args.output_path
        self.save_model_path = os.path.join(self.output_path, 'models/')
        self.save_logs_path = os.path.join(self.output_path, 'logs/')
        self._init_device()

        self.num_classes = num_classes

        self._set_task()

        # self.Return_B4_BN = args.Return_B4_BN
        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size,download_root="/data0/hzy_log/pretrain_cache/clip",prj_output_dim=self.args.prj_output_dim,pooling=self.args.pooling)
        self.embed_dim = base_cfg['embed_dim']
        if args.pretrain_choice == 'RN50':
            # 复制conv1...的权重到conv1_...
            print("copy conv1 weight to conv1_")
            self.base_model.visual.conv1_.load_state_dict(self.base_model.visual.conv1.state_dict())
            # self.base_model.visual.conv2_.load_state_dict(self.base_model.visual.conv2.state_dict())
            # self.base_model.visual.conv3_.load_state_dict(self.base_model.visual.conv3.state_dict())
    

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / args.temperature))  # 0.07
        # self.logit_scale = torch.ones([]) * np.log(1 / args.temperature)  # 0.07

        # Fusion way definition
        # if args.fusion_way == 'weight add':
        #     self.gate = nn.Parameter(torch.FloatTensor(2))
        #     nn.init.constant_(self.gate, 0.5)
        # if args.fusion_way == 'concat':
        #     scale = 512**-0.5
        #     proj_std = scale * ((2 * 4)**-0.5)
        #     self.dim_conv = nn.Linear(512*2,512)
        #     nn.init.normal_(self.dim_conv.weight, std=proj_std)
        if 'attention' in args.fusion_way:
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)
        if 'global' in args.fusion_way:
            # self.dim_conv = nn.Linear(512*2,512)
            self.global_attn_s = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.global_attn_t = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
             # init cross attn
            scale = 512**-0.5
            proj_std = scale * ((2 * 4)**-0.5)
            attn_std = scale
            # fc_std = (2 * 512)**-0.5
            nn.init.normal_(self.global_attn_s.in_proj_weight, std=attn_std)
            nn.init.normal_(self.global_attn_s.out_proj.weight, std=proj_std)
             # init cross attn
            nn.init.normal_(self.global_attn_t.in_proj_weight, std=attn_std)
            nn.init.normal_(self.global_attn_t.out_proj.weight, std=proj_std)
            # nn.init.normal_(self.dim_conv.weight, std=proj_std)
        # if 'concat' in args.fusion_way:
        #     self.cross_modal_transformer = Transformer(width=self.embed_dim,
        #                                                layers=args.cmt_depth,
        #                                                heads=self.embed_dim //
        #                                                64)
        if 'attention' in args.fusion_way and 'global' not in args.fusion_way:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            # self.cross_modal_transformer = Transformer(width=self.embed_dim,
            #                                            layers=args.cmt_depth,
            #                                            heads=self.embed_dim //
            # #                                            64)
            scale = self.embed_dim ** -0.5
            # self.pos_embedding = nn.Parameter(scale * torch.randn(self.embed_dim))
            

            proj_std = scale * ((2 * args.cmt_depth)**-0.5)
            attn_std = scale
            # fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            # for block in self.cross_modal_transformer.resblocks:
            #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

        # Loss definition
        self.classifier = Classifier(self.num_classes,self.embed_dim,args.Return_B4_BN,args.uni_BN,args.joint_mode)
        self.pid_criterion = nn.CrossEntropyLoss()
        self.tri_criterion = TripletLoss_WRT()
        self.wrt_local = TripletLoss_WRT_local()

    def _init_device(self):
        self.device = torch.device(
            'cuda:{}'.format(self.args.gpu_id) if torch.cuda.is_available() else 'cpu')
        print('Model is using device: {}'.format(self.device))

    def set_train(self):
        self.train()
        self.training = True

    def set_eval(self):
        self.eval()
        self.training = False
    
    def save_model(self, save_epoch, is_best, mode='Fusion'): # mode = ['IR', 'Fusion', 'Text'] or their composition
        if is_best:
            model_file_path = os.path.join(self.save_model_path, f'model_{mode}_{save_epoch}.pth')
            if self.args.DataParallel:
                torch.save(self.module.state_dict(), model_file_path)
            else:
                torch.save(self.state_dict(), model_file_path)

        if self.max_save_model_num > 0:
            root, _, files = os_walk(self.save_model_path)
            if mode in ['Fusion', 'IR', 'Text']:
                valid_files = []
                for i,file in enumerate(files):
                    if ('.pth' in file) and (mode in file):
                        valid_files.append(file)
                if len(valid_files) > self.max_save_model_num:
                    file_iters = sorted([int(file.replace('.pth', '').split('_')[-1]) for file in valid_files], reverse=False)
                    model_file_path = os.path.join(root, f'model_{mode}_{file_iters[0]}.pth')
                    os.remove(model_file_path)
            else:
                raise ValueError("savinf mode must be in ['Fusion', 'IR', 'Text']")


    def resume_last_model(self,mode='Fusion'):
        root, _, files = os_walk(self.save_model_path)
        valid_files = []
        for file in files:
            if ('pth' in file) and (mode in file):
                valid_files.append(file)

        if len(files) > 0:
            indexes = []
            for file in valid_files:
                indexes.append(int(file.replace('.pth', '').split('_')[-1]))
            indexes = sorted(list(set(indexes)), reverse=False)
            self.resume_model(indexes[-1],mode)
            start_train_epoch = indexes[-1]
            return start_train_epoch
        else:
            return 0

    def resume_model(self, resume_epoch, mode='Fusion'):
        model_path = os.path.join(self.save_model_path, f'model_{mode}_{resume_epoch}.pth')
        print('Resume model from {}'.format(model_path))
        self.load_state_dict(torch.load(model_path), strict=False)
        print('Successfully resume model from {}'.format(model_path))


    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split(',')]
        print(f'Training Model with {self.current_task} tasks')
    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = q + x # residual connection (invalid for mcq and mcqmlm, valid for mlm)
        # x = x.permute(1, 0, 2)  # NLD -> LND
        # x = self.cross_modal_transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x
    
    def global_former_s(self, q, k, v):
        x = self.global_attn_s(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        
        x = q + x # residual connection (invalid for mcq and mcqmlm, valid for mlm)
        # x = x.permute(1, 0, 2)  # NLD -> LND
        # x = self.cross_modal_transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def global_former_t(self, q, k, v):
        x = self.global_attn_t(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = q + x # residual connection (invalid for mcq and mcqmlm, valid for mlm)
        # x = x.permute(1, 0, 2)  # NLD -> LND
        # x = self.cross_modal_transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image_featmap(self, image, mode=None): 
        x = self.base_model.encode_image(image,mode=mode)
        return x
        # return x.float() # for CLIP ResNet visual model

    def encode_text_featmap(self, text):
        x = self.base_model.encode_text(text)
        return x #[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()
    
    def encode_image_feat(self, image, mode=None): # return [B, 512]
        x = self.base_model.encode_image(image,mode=mode)
        if 'RN' in self.args.pretrain_choice:
            return self.base_model.visual.__getattr__(self.args.pooling)(x).float().squeeze()
        elif "ViT" in self.args.pretrain_choice:
            return x[:, 0, :].float()
        else:
            raise ValueError("pretrain_choice must be in ['RN50', 'RN50_ORI', 'ViT-B/16']")

    def encode_text_feat(self, text): # return [B, 512]
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()
    
    def encode_fusion(self, text, ir, mode='ir'):
        # 获取 id 形式的文本原始数据
        caption_ids = text 
        # 获取文本Tensor特征
        text = self.encode_text_featmap(text)
        # 获取IR图像Tensor特征
        ir = self.encode_image_featmap(ir,mode=mode)
        # 获取融合后的特征
        x = self.fusion_layer(text,ir,caption_ids,pa=self.args.pa, way=self.args.fusion_way)
        return x.float()
    
    def encode_filtered_fusion(self, text, filter, ir):
        # 获取 id 形式的文本原始数据
        caption_ids = text 
        filter_caption_ids = filter
        # 获取文本Tensor特征
        text_feat = self.encode_text_feat(caption_ids)
        # 获取filter Tensor特征
        filter_text_feat = self.encode_text_feat(filter_caption_ids)
        # 获取IR图像Tensor特征
        ir = self.encode_image_feat(ir,mode='ir')
        # 获取融合后的特征
        x = ir + text_feat - filter_text_feat
        return x.float()

    def text_fusion_layer(self, text_rgb_map, text_ir_map, caption_rgb_ids, caption_ir_ids, way='add'):
        if way == 'norm_add':
            text_rgb_feats = text_rgb_map[torch.arange(text_rgb_map.shape[0]), caption_rgb_ids.argmax(dim=-1)]
            text_ir_feats = text_ir_map[torch.arange(text_ir_map.shape[0]), caption_ir_ids.argmax(dim=-1)]
            f_text_feats = text_rgb_feats/text_rgb_feats.norm(dim=-1,keepdim=True) + text_ir_feats/text_ir_feats.norm(dim=-1,keepdim=True)

        if way == 'add':
            text_rgb_feats = text_rgb_map[torch.arange(text_rgb_map.shape[0]), caption_rgb_ids.argmax(dim=-1)]
            text_ir_feats = text_ir_map[torch.arange(text_ir_map.shape[0]), caption_ir_ids.argmax(dim=-1)]
            f_text_feats = text_rgb_feats + text_ir_feats
        
        if way == 'cross_attention':
            text_rgb_feats = text_rgb_map[torch.arange(text_rgb_map.shape[0]), caption_rgb_ids.argmax(dim=-1)]
            text_ir_feats = text_ir_map[torch.arange(text_ir_map.shape[0]), caption_ir_ids.argmax(dim=-1)]
            f_text_feats = (self.cross_former(text_rgb_feats.unsqueeze(1),text_ir_map,text_ir_map) + self.cross_former(text_ir_feats.unsqueeze(1),text_rgb_map,text_rgb_map))
            f_text_feats = f_text_feats.squeeze(1).contiguous()
        
        if way == 'attention_rgb_text':
            text_rgb_feats = text_rgb_map[torch.arange(text_rgb_map.shape[0]), caption_rgb_ids.argmax(dim=-1)]
            f_text_feats = self.cross_former(text_rgb_feats.unsqueeze(1),text_ir_map,text_ir_map).squeeze(1).contiguous()
        
        if way == 'attention_ir_text':
            text_ir_feats = text_ir_map[torch.arange(text_ir_map.shape[0]), caption_ir_ids.argmax(dim=-1)]
            f_text_feats = self.cross_former(text_ir_feats.unsqueeze(1),text_rgb_map,text_rgb_map).squeeze(1).contiguous()
        
        if way == 'global_attention':
            f_text_feats = (self.global_former_s(text_rgb_map,text_ir_map,text_ir_map)[torch.arange(text_rgb_map.shape[0]), caption_rgb_ids.argmax(dim=-1)] +\
                             self.global_former_t(text_ir_map,text_rgb_map,text_rgb_map)[torch.arange(text_ir_map.shape[0]), caption_ir_ids.argmax(dim=-1)])
            f_text_feats = f_text_feats.squeeze(1).contiguous()

        if way == 'global_attention_rgb_text':
            f_text_feats = self.global_former_s(text_rgb_map,text_ir_map,text_ir_map)[torch.arange(text_rgb_map.shape[0]), caption_rgb_ids.argmax(dim=-1)].squeeze(1).contiguous()
        
        if way == 'global_attention_ir_text':
            f_text_feats = self.global_former_t(text_ir_map,text_rgb_map,text_rgb_map)[torch.arange(text_ir_map.shape[0]), caption_ir_ids.argmax(dim=-1)].squeeze(1).contiguous()

        return f_text_feats

    def fusion_layer(self, text_map, ir_map, caption_ids, pa=0.1, way='add'):
        text_feats = text_map[torch.arange(text_map.shape[0]), caption_ids.argmax(dim=-1)]
        if 'RN' in self.args.pretrain_choice:
            ir_feats = self.base_model.visual.__getattr__(self.args.pooling)(ir_map).squeeze()
        if 'VIT' in self.args.pretrain_choice:
            ir_feats = ir_map[:, 0, :]
        if way == 'norm_add':
            t_norm = text_feats.norm(dim=-1,keepdim=True)
            ir_norm = ir_feats/ir_feats.norm(dim=-1,keepdim=True)
            f_feats = (text_feats/t_norm + ir_feats/ir_norm)/2 * ir_norm
        elif way == 'weight_add': # feat and feat
            f_feats = self.gate[0] * text_feats + self.gate[1] * ir_feats
        elif way == 'cross_attention':
            f_feats = (self.cross_former(text_feats.unsqueeze(1),ir_map,ir_map) + self.cross_former(ir_feats.unsqueeze(1),text_map,text_map))
            f_feats = f_feats.squeeze(1).contiguous()
        elif way == 'cross_attention_text':
            # f_feats = (self.cross_former(text,sketch,sketch)[:, 0, :] + self.cross_former(sketch,text,text)[torch.arange(text.shape[0]), caption_ids.argmax(dim=-1)])
            f_feats = self.cross_former(ir_feats.unsqueeze(1),text_map,text_map).squeeze(1).contiguous()
        elif way == 'cross_attention_ir':
            # f_feats = (self.cross_former(text,sketch,sketch)[:, 0, :] + self.cross_former(sketch,text,text)[torch.arange(text.shape[0]), caption_ids.argmax(dim=-1)])
            f_feats = self.cross_former(text_feats.unsqueeze(1),ir_map,ir_map).squeeze(1).contiguous()
        elif way == 'parameter_add':
            f_feats = (1-pa)*text_feats + pa*ir_feats
        elif way == 'concat':
            f_feats = self.dim_conv(torch.cat((text_feats, ir_feats),dim=1))
        elif way == 'concat_transformer':
            l_t = text_map.size(1)
            f_feats = self.cross_modal_transformer(torch.cat((text_map,ir_map),dim=1))
            f_feats = f_feats[:,l_t:,:][:, 0, :] + f_feats[:,:l_t,:][torch.arange(text_map.shape[0]), caption_ids.argmax(dim=-1)]
        elif way == 'concat_transformer-i':
            l_t = text_map.size(1)
            f_feats = self.cross_modal_transformer(torch.cat((text_map,ir_map),dim=1))
            f_feats = f_feats[:,l_t:,:][:, 0, :]
        elif way == 'concat_transformer-t':
            l_t = text_map.size(1)
            f_feats = self.cross_modal_transformer(torch.cat((text_map,ir_map),dim=1))
            f_feats = f_feats[:,:l_t,:][torch.arange(text_map.shape[0]), caption_ids.argmax(dim=-1)]
        else:
            f_feats = text_feats + ir_feats
            
        return f_feats.float()

    def forward(self, batch_dict, mode=None):
        # get data
        rgb_imgs0 = batch_dict['img_rgb_ori']
        rgb_imgs1 = batch_dict['img_rgb_aug']
        ir_imgs = batch_dict['img_ir']
        label_rgb = batch_dict['target_rgb']
        label_ir = batch_dict['target_ir']

        # init return dict
        ret = dict()

        # get feature map
        if self.args.Fix_Visual:
            with torch.no_grad():
                image_feats_map = self.base_model.encode_image(torch.cat((rgb_imgs0,rgb_imgs1,ir_imgs),dim=0), mode)
        else:
            image_feats_map = self.base_model.encode_image(torch.cat((rgb_imgs0,rgb_imgs1,ir_imgs),dim=0), mode)
        b = ir_imgs.size(0)


        if 'RN' in self.args.pretrain_choice:     # img: [64 + 32, 1024] text:[64, 77, 1024]
            rgb_feats_map = image_feats_map[:int(2*b)]
            ir_feats_map = image_feats_map[int(2*b):]
            if self.args.Fix_Visual:
                with torch.no_grad():
                    rgb_feats = self.base_model.visual.__getattr__(self.args.pooling)(rgb_feats_map).float().squeeze()
                    ir_feats = self.base_model.visual.__getattr__(self.args.pooling)(ir_feats_map).float().squeeze()
            else:
                rgb_feats = self.base_model.visual.__getattr__(self.args.pooling)(rgb_feats_map).float().squeeze()
                ir_feats = self.base_model.visual.__getattr__(self.args.pooling)(ir_feats_map).float().squeeze()

        elif "ViT" in self.args.pretrain_choice:  # img: [64 + 32, 193, 512] text:[64, 77, 512]
            rgb_feats_map = image_feats_map[:int(2*b),:,:]
            ir_feats_map = image_feats_map[int(2*b):,:,:] 
            rgb_feats = rgb_feats_map[:, 0, :].float()
            ir_feats = ir_feats_map[:, 0, :].float()
        else:
            raise ValueError("pretrain_choice must be in ['RN50', 'RN50_ORI', 'ViT-B/16']")

        
        logit_scale = self.logit_scale.exp()
        ret.update({'temperature': 1 / logit_scale})

           
        loss_list = self.args.loss_names.split(',')

        if self.args.training_mode == 'RGB_IR_Text': # 如果有文本信息辅助
            if self.args.fusion_way in ['norm_add', 'add', 'cross_attention', 'attention_rgb_text', 'attention_ir_text', 'global_attention', 'global_attention_rgb_text', 'global_attention_ir_text']:
                
                # 获取rgb图像特征
                ori_vi_feats = rgb_feats[:int(b)]
                aug_vi_feats = rgb_feats[int(b):]
                
                if self.args.joint_mode == 'dual_text':
                    text_rgb = batch_dict['text_rgb']
                    text_ir = batch_dict['text_ir']
                    text_rgb_feats_map = self.base_model.encode_text(text_rgb).detach()
                    text_ir_feats_map = self.base_model.encode_text(text_ir).detach()
                    t_feats = self.text_fusion_layer(text_rgb_feats_map,text_ir_feats_map,text_rgb,text_ir,way=self.args.fusion_way)
                    if 'id' in loss_list:
                        # get labels
                        pids = torch.cat([label_rgb,label_rgb,label_ir], dim=0)
                        img_feats = torch.cat((rgb_feats, ir_feats),dim=0)
                        _,img_scores = self.classifier(img_feats)
                        ret.update({'id_loss':(self.pid_criterion(img_scores, pids))*self.args.id_loss_weight})
                        img_acc = (img_scores.max(1)[1] == pids).float().mean()
                        ret.update({'acc': img_acc})
                    
                    if 'wrt' in loss_list:
                        ret.update({'wrt_loss':(self.tri_criterion(img_feats, pids))*self.args.wrt_loss_weight})
                    
                    if 'i2t' in loss_list:
                        ret.update({'i2t_loss':L_i2t(ir_feats,t_feats,logit_scale) + \
                                    0.5*(L_i2t(ori_vi_feats,t_feats,logit_scale) + L_i2t(aug_vi_feats,t_feats,logit_scale))})
                    
                    if 't2i' in loss_list:
                        ret.update({'t2i_loss':L_t2i(ir_feats,t_feats,logit_scale,label_ir) + \
                                    0.5*(L_t2i(ori_vi_feats,t_feats,logit_scale,label_rgb) + L_t2i(aug_vi_feats,t_feats,logit_scale,label_rgb))})
                    
                   
                elif self.args.joint_mode == 'uni':
                    text_rgb = batch_dict['text_rgb']
                    text_rgb_feats_map = self.base_model.encode_text(text_rgb)

                    # 获取融合后的特征
                    if self.args.Feat_Filter:
                        text_filter = batch_dict['text_ir']
                        text_filter_feats = self.encode_text_feat(text_filter)
                        t_feats = text_rgb_feats_map[torch.arange(text_rgb_feats_map.shape[0]), text_rgb.argmax(dim=-1)]
                        f_feats = (ir_feats + t_feats - text_filter_feats).squeeze()
                        t_feats = t_feats.float()

                    else:
                        f_feats = self.fusion_layer(text_rgb_feats_map, ir_feats_map, text_rgb, pa=self.args.pa, way=self.args.fusion_way).squeeze()
                        # 获取文本特征
                        t_feats = text_rgb_feats_map[torch.arange(text_rgb_feats_map.shape[0]), text_rgb.argmax(dim=-1)].float() #[64, 512]
                    
                    # # uni_id
                    # uni_pids = torch.cat([label_rgb,label_rgb,label_ir,label_ir,label_ir], dim=0)
                    # all_feats = torch.cat((ori_vi_feats, aug_vi_feats, ir_feats, f_feats, t_feats), dim=0)
                    if "id" in loss_list:
                        # uni_id
                        uni_pids = torch.cat([label_rgb,label_rgb,label_ir,label_ir,label_ir], dim=0)
                        all_feats = torch.cat((ori_vi_feats, aug_vi_feats, ir_feats, f_feats, t_feats), dim=0)
                        _, all_feat_scores = self.classifier(all_feats)
                        ret.update({'uni_id_loss':(self.pid_criterion(all_feat_scores, uni_pids))*self.args.id_loss_weight})
                        feat_acc = (all_feat_scores.max(1)[1] == uni_pids).float().mean()
                        ret.update({'acc': feat_acc})
                    
                    if "id_woir" in loss_list:
                        uni_woir_pids = torch.cat([label_rgb,label_rgb,label_ir,label_ir],dim=0)
                        uni_woir_feats = torch.cat([rgb_feats,f_feats,t_feats],dim=0)   
                        _, img_scores = self.classifier(uni_woir_feats)
                        ret.update({'uni_id_woir_loss':(self.pid_criterion(img_scores, uni_woir_pids))*self.args.id_loss_weight})
                        feat_acc = (img_scores.max(1)[1] == uni_woir_pids).float().mean()
                        ret.update({'acc': feat_acc})
                    
                    if "wrt" in loss_list:
                        # uni_wrt
                        ret.update({'uni_wrt_loss':(self.tri_criterion(all_feats, uni_pids))*self.args.wrt_loss_weight})
                    
                    if "wrt_woir" in loss_list:
                        # uni_wrt
                        ret.update({'uni_wrt_woir_loss':(self.tri_criterion(uni_woir_feats, uni_woir_pids))*self.args.wrt_loss_weight})

                    if "orth" in loss_list:
                        # uni_orth
                        ret.update({'uni_orth_loss':objectives.orthogonal_loss(ir_feats, t_feats, text_filter_feats)})
                    
                    if "orth2" in loss_list:
                        # uni_orth
                        ret.update({'uni_orth2_loss':objectives.orthogonal_loss2(ir_feats, t_feats, text_filter_feats)})

                    if "T2I_Regular" in loss_list:
                        ret.update({"T2I_Regular_loss":kl_align_loss(ir_feats,f_feats,t_feats,logit_scale,mode='T2I')})
                        
                    if "I2T_Regular" in loss_list:
                        ret.update({"I2T_Regular_loss":kl_align_loss(ir_feats,f_feats,t_feats,logit_scale,mode='I2T')})
                    

                elif self.args.joint_mode == 'ir_crossfusion':

                    text_rgb = batch_dict['text_rgb']
                    text_rgb_feats_map = self.base_model.encode_text(text_rgb)
                    # 获取融合后的特征
                    if self.args.Feat_Filter:
                        text_filter = batch_dict['text_ir']
                        text_filter_feats = self.encode_text_feat(text_filter)
                        f_feats = (ir_feats + text_rgb_feats_map[torch.arange(text_rgb_feats_map.shape[0]), text_rgb.argmax(dim=-1)] - text_filter_feats).squeeze()

                    else:
                        f_feats = self.fusion_layer(text_rgb_feats_map, ir_feats_map, text_rgb, pa=self.args.pa, way=self.args.fusion_way).squeeze()
                                       
                    if 'id' in loss_list:
                        # get labels
                        pids = torch.cat([label_rgb,label_rgb,label_ir], dim=0)
                        img_feats = torch.cat((ori_vi_feats, aug_vi_feats, f_feats),dim=0)
                        _, img_scores = self.classifier(img_feats)
                        ret.update({'id_loss':(self.pid_criterion(img_scores, pids))*self.args.id_loss_weight})
                        img_acc = (img_scores.max(1)[1] == pids).float().mean()
                        ret.update({'acc': img_acc})

                    if 'wrt' in loss_list:
                        ret.update({'wrt_loss':(self.tri_criterion(img_feats, pids))*self.args.wrt_loss_weight})

                    # if "T2I_Regular" in loss_list:
                    #     ret.update({"T2I_Regular_loss":kl_align_loss(ir_feats,f_feats,t_feats,logit_scale,mode='T2I')})

                    # if "I2T_Regular" in loss_list:
                    #     ret.update({"I2T_Regular_loss":kl_align_loss(ir_feats,f_feats,t_feats,logit_scale,mode='I2T')})
                    
                        
                elif self.args.joint_mode == 'ir_selffusion':
                    
                    text_rgb_w = batch_dict['text_rgb_w']
                    text_rgb_w_feats_map = self.base_model.encode_text(text_rgb_w)
                    
                    # 获取融合后的特征
                    f_feats = self.fusion_layer(text_rgb_w_feats_map, ir_feats_map, text_rgb_w, pa=self.args.pa, way=self.args.fusion_way).squeeze()                    
                    
                    if 'id' in loss_list:
                        # get labels
                        pids = torch.cat([label_rgb,label_rgb,label_ir], dim=0)
                        img_feats = torch.cat((rgb_feats, f_feats), dim=0)
                        _, img_scores = self.classifier(img_feats)
                        ret.update({'id_loss':(self.pid_criterion(img_scores, pids))*self.args.id_loss_weight})
                        img_acc = (img_scores.max(1)[1] == pids).float().mean()
                        ret.update({'acc': img_acc})

                    if 'wrt' in loss_list:
                        ret.update({'wrt_loss':(self.tri_criterion(img_feats, pids))*self.args.wrt_loss_weight})

                    # if "T2I_Regular" in loss_list:
                    #     ret.update({"T2I_Regular_loss":kl_align_loss(ir_feats,f_feats,t_feats,logit_scale,mode='T2I')})

                    # if "I2T_Regular" in loss_list:
                    #     ret.update({"I2T_Regular_loss":kl_align_loss(ir_feats,f_feats,t_feats,logit_scale,mode='I2T')})
                    

                elif self.args.joint_mode == 'rgb_selffusion':
                    
                    text_ir = batch_dict['text_ir']
                    text_ir_feats_map = self.base_model.encode_text(text_ir)
                    
                    # 获取融合后的特征
                    f_feats = self.fusion_layer(torch.cat([text_ir_feats_map,text_ir_feats_map],dim=0), rgb_feats_map, torch.cat([text_ir,text_ir],dim=0), pa=self.args.pa, way=self.args.fusion_way).squeeze()
                    
                    if 'id' in loss_list:
                        # get labels
                        pids = torch.cat([label_rgb,label_rgb,label_ir], dim=0)
                        img_feats = torch.cat((f_feats, ir_feats),dim=0)
                        _,img_scores = self.classifier(img_feats)
                        ret.update({'id_loss':(self.pid_criterion(img_scores, pids))*self.args.id_loss_weight})
                        img_acc = (img_scores.max(1)[1] == pids).float().mean()
                        ret.update({'acc': img_acc})

                    if 'wrt' in loss_list:
                        ret.update({'wrt_loss':(self.tri_criterion(img_feats, pids))*self.args.wrt_loss_weight})

                    # if "T2I_Regular" in loss_list:
                    #     ret.update({"T2I_Regular_loss":kl_align_loss(ir_feats,f_feats,t_feats,logit_scale,mode='T2I')})

                    # if "I2T_Regular" in loss_list:
                    #     ret.update({"I2T_Regular_loss":kl_align_loss(ir_feats,f_feats,t_feats,logit_scale,mode='I2T')})

                else:
                    raise ValueError("joint mode must be in ['uni', 'ir_crossfusion', 'ir_selffusion', 'dual_selffusion']")
            else:  # 如果融合方式没有被定义
                raise NotImplementedError("Fusion way must be in ['norm_add', 'add', 'cross_attention', 'attention_rgb_text', 'attention_ir_text', 'global_attention', 'global_attention_rgb_text', 'global_attention_ir_text']")

        elif self.args.training_mode == "RGB_IR": 
            pids = torch.cat([label_rgb,label_rgb,label_ir], dim=0)
            img_feats = torch.cat((rgb_feats, ir_feats), dim=0)
            if 'id' in loss_list:
                _, scores = self.classifier(img_feats)
                ret.update({'id_loss':(self.pid_criterion(scores, pids))*self.args.id_loss_weight})
                acc = (scores.max(1)[1] == pids).float().mean()
                ret.update({'acc': acc})

            if 'wrt' in loss_list:
                ret.update({'wrt_loss':(self.tri_criterion(img_feats, pids))*self.args.wrt_loss_weight})

        elif self.args.training_mode == "RGB_Text":
            text_rgb = batch_dict['text_rgb']
            text_rgb_feats_map = self.base_model.encode_text(text_rgb)
            t_feats = text_rgb_feats_map[torch.arange(text_rgb_feats_map.shape[0]), text_rgb.argmax(dim=-1)].float()
            img_text_feats = torch.cat((rgb_feats, t_feats), dim=0)
            pids = torch.cat([label_rgb,label_rgb,label_ir], dim=0)
            if 'id' in loss_list:
                _, scores = self.classifier(img_text_feats)
                ret.update({'id_loss':(self.pid_criterion(scores, pids))*self.args.id_loss_weight})
                acc = (scores.max(1)[1] == pids).float().mean()
                ret.update({'acc': acc})

            if 'wrt' in loss_list:
                ret.update({'wrt_loss':(self.tri_criterion(img_text_feats, pids))*self.args.wrt_loss_weight})
        else:
            raise ValueError("training mode must be in ['RGB_IR_Text', 'RGB_IR', 'RGB_Text']")
 
        return ret



def build_model(config):
    model = CLIP2ReID(config, num_classes=config.pid_num)
    return model


