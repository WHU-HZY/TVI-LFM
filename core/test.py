
import numpy as np
import torch
from torch.autograd import Variable
from tools import eval_regdb, eval_sysu, eval_llcm
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def test(base, loader, config, device):
    embed_dim = base.embed_dim
    base.set_eval()
    print('Extracting Query Feature...')
    ptr = 0
    print('Test Mode: ', config.test_modality)
    result_dict = dict()
    
    assert loader.dataset in ['sysu', 'regdb', 'llcm'], 'Invalid dataset!'
    assert 'IR' in config.test_modality or 'Fusion' in config.test_modality or 'Text' in config.test_modality, 'Invalid test modality!'

    if loader.dataset == 'regdb':
        query_feat_ir_list = []
        query_feat_fusion_list = []
        query_feat_text_list = []
        for i in range(config.eval_num_regdb):
            ptr = 0
            query_loader = loader.query_loaders[i]
            if 'IR' in config.test_modality:
                query_feat_ir = np.zeros((loader.n_query, embed_dim))
            if 'Fusion' in config.test_modality:
                query_feat_fusion = np.zeros((loader.n_query, embed_dim))
                if config.CAT_EVAL:
                    #####################
                    query_feat_ir_f = np.zeros((loader.n_query, embed_dim))
                    #####################
                    query_feat_text_f = np.zeros((loader.n_query, embed_dim))
                    #####################
            if 'Text' in config.test_modality:
                query_feat_text = np.zeros((loader.n_query, embed_dim))
            with torch.no_grad():
                # for batch_idx, (input, text, label) in enumerate(loader.query_loader):
                for batch_idx, batch_dict in enumerate(query_loader):
                    input = batch_dict['img']
                    # label = batch_dict['target']

                    batch_num = input.size(0)
                    input = Variable(input.to(device))
                    
                    if  'Fusion' or 'Text' in config.test_modality:
                        text = batch_dict['text']
                        text = text.to(device).long()
                        
                    # get the feature of the last layer 
                    if 'IR' in config.test_modality:
                        feat_ir_map = base.encode_image_featmap(input,'ir') # IR mode
                        if config.Fix_Visual:
                            feat_ir = base.backup_pool(feat_ir_map).squeeze()
                            feat_ir = base.backup_classifier(feat_ir)
                        else:
                            feat_ir = base.base_model.visual.__getattr__(config.pooling)(feat_ir_map).squeeze() # IR mode
                            feat_ir = base.classifier(feat_ir)
                        query_feat_ir[ptr:ptr + batch_num, :] = feat_ir.detach().cpu().numpy()
                    if 'Fusion' in config.test_modality:
                        if config.Feat_Filter:
                            text_filter = batch_dict['text_filter']
                            text_filter = text_filter.to(device).long()
                            feat_fusion = base.encode_filtered_fusion(text,text_filter,input)  # Fusion mode  rgbtext + irimage
                        else:
                            feat_fusion = base.encode_fusion(text,input,'ir')  # Fusion mode  rgbtext + irimage
                            
                        feat_fusion = base.classifier(feat_fusion,"Fusion") # [64, 512] -> [64, 512]
                        query_feat_fusion[ptr:ptr + batch_num, :] = feat_fusion.detach().cpu().numpy()
                        if config.CAT_EVAL:
                            ##################################################
                            feat_text = base.encode_text_feat(text)
                            feat_text = base.classifier(feat_text)
                            query_feat_text_f[ptr:ptr + batch_num, :] = feat_text.detach().cpu().numpy()
                            ##################################################
                            feat_ir = base.encode_image_feat(input,'ir')
                            feat_ir = base.classifier(feat_ir)
                            query_feat_ir_f[ptr:ptr + batch_num, :] = feat_ir.detach().cpu().numpy()
                            ##################################################

                    if 'Text' in config.test_modality:
                        feat_text = base.encode_text_feat(text)
                        feat_text = base.classifier(feat_text,'Text')
                        query_feat_text[ptr:ptr + batch_num, :] = feat_text.detach().cpu().numpy()

                    ptr = ptr + batch_num
                if 'IR' in config.test_modality:
                    query_feat_ir_list.append(query_feat_ir)
                if 'Fusion' in config.test_modality:   
                    query_feat_fusion_list.append(query_feat_fusion)
                if 'Text' in config.test_modality:
                    query_feat_text_list.append(query_feat_text)
                

    else:
        if 'IR' in config.test_modality:
            query_feat_ir = np.zeros((loader.n_query, embed_dim))
        if 'Fusion' in config.test_modality:
            query_feat_fusion = np.zeros((loader.n_query, embed_dim))
            if config.CAT_EVAL:
                #####################
                query_feat_ir_f = np.zeros((loader.n_query, embed_dim))
                #####################
                query_feat_text_f = np.zeros((loader.n_query, embed_dim))
                #####################
        if 'Text' in config.test_modality:
            query_feat_text = np.zeros((loader.n_query, embed_dim))
        with torch.no_grad():
            # for batch_idx, (input, text, label) in enumerate(loader.query_loader):
            for batch_idx, batch_dict in enumerate(loader.query_loader):
                input = batch_dict['img']
                # label = batch_dict['target']

                batch_num = input.size(0)
                input = Variable(input.to(device))
                
                if  'Fusion' or 'Text' in config.test_modality:
                    text = batch_dict['text']
                    text = text.to(device).long()
                    
                # get the feature of the last layer 
                if 'IR' in config.test_modality:
                    feat_ir_map = base.encode_image_featmap(input,'ir') # IR mode
                    if config.Fix_Visual:
                        feat_ir = base.backup_pool(feat_ir_map).squeeze()
                        feat_ir = base.backup_classifier(feat_ir)
                    else:
                        feat_ir = base.base_model.visual.__getattr__(config.pooling)(feat_ir_map).squeeze() # IR mode
                        feat_ir = base.classifier(feat_ir)
                    query_feat_ir[ptr:ptr + batch_num, :] = feat_ir.detach().cpu().numpy()
                if 'Fusion' in config.test_modality:
                    if config.Feat_Filter:
                        text_filter = batch_dict['text_filter']
                        text_filter = text_filter.to(device).long()
                        feat_fusion = base.encode_filtered_fusion(text,text_filter,input)  # Fusion mode  rgbtext + irimage
                    else:
                        feat_fusion = base.encode_fusion(text,input,'ir')  # Fusion mode  rgbtext + irimage
                        
                    feat_fusion = base.classifier(feat_fusion,"Fusion") # [64, 512] -> [64, 512]
                    query_feat_fusion[ptr:ptr + batch_num, :] = feat_fusion.detach().cpu().numpy()
                    if config.CAT_EVAL:
                        ##################################################
                        feat_text = base.encode_text_feat(text)
                        feat_text = base.classifier(feat_text)
                        query_feat_text_f[ptr:ptr + batch_num, :] = feat_text.detach().cpu().numpy()
                        ##################################################
                        feat_ir = base.encode_image_feat(input,'ir')
                        feat_ir = base.classifier(feat_ir)
                        query_feat_ir_f[ptr:ptr + batch_num, :] = feat_ir.detach().cpu().numpy()
                        ##################################################
                if 'Text' in config.test_modality:
                    feat_text = base.encode_text_feat(text)
                    feat_text = base.classifier(feat_text,'Text')
                    query_feat_text[ptr:ptr + batch_num, :] = feat_text.detach().cpu().numpy()

                ptr = ptr + batch_num



    print('Extracting Gallery Feature...')

    if loader.dataset == 'sysu':
        all_cmc_ir = 0
        all_mAP_ir = 0
        all_mINP_ir = 0
        all_cmc_fusion = 0
        all_mAP_fusion = 0
        all_mINP_fusion = 0
        all_cmc_text = 0
        all_mAP_text = 0
        all_mINP_text = 0
        for i in range(10):
            ptr = 0
            gall_loader = loader.gallery_loaders[i]
            if 'IR' in config.test_modality and config.Fix_Visual:
                    gall_feat_for_IR = np.zeros((loader.n_gallery, embed_dim))
            if 'IR' in config.test_modality or 'Fusion' in config.test_modality or 'Text' in config.test_modality:
                gall_feat = np.zeros((loader.n_gallery, embed_dim))
            with torch.no_grad():
                # for batch_idx, (input, text, label) in enumerate(gall_loader):
                for batch_idx, batch_dict in enumerate(gall_loader):
                    input = batch_dict['img']
                    # label = batch_dict['target']

                    batch_num = input.size(0)
                    input = Variable(input.to(device))

                    # get the feature of the last layer 
                    feat_map = base.encode_image_featmap(input,'rgb')
                    if 'IR' in config.test_modality and config.Fix_Visual:
                        feat_for_IR = base.backup_pool(feat_map).squeeze()
                        feat_for_IR = base.backup_classifier(feat_for_IR)
                        gall_feat_for_IR[ptr:ptr + batch_num, :] = feat_for_IR.detach().cpu().numpy()
                    if 'IR' in config.test_modality or 'Text' in config.test_modality or 'Fusion' in config.test_modality:
                        feat = base.base_model.visual.__getattr__(config.pooling)(feat_map).squeeze()
                        feat = base.classifier(feat)
                        gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                    else: 
                        ValueError('Error: test_modality not found!')

                    gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                    ptr = ptr + batch_num
                    
            if 'IR' in config.test_modality:
                if config.Fix_Visual:
                    distmat_ir = np.matmul(query_feat_ir, np.transpose(gall_feat_for_IR))
                else:
                    distmat_ir = np.matmul(query_feat_ir, np.transpose(gall_feat))
                cmc_ir, mAP_ir, mINP_ir = eval_sysu(-distmat_ir, loader.query_label, loader.gall_label, loader.query_cam,
                                        loader.gall_cam)
                all_cmc_ir += cmc_ir
                all_mAP_ir += mAP_ir
                all_mINP_ir += mINP_ir
            if 'Fusion' in config.test_modality:
                if config.CAT_EVAL:
                    distmat_ir_f = np.matmul(query_feat_ir_f, np.transpose(gall_feat))
                    distmat_text_f = np.matmul(query_feat_text_f, np.transpose(gall_feat))
                    distmat_fusion = np.matmul(query_feat_fusion, np.transpose(gall_feat)) + distmat_ir_f + distmat_text_f
                else:
                    distmat_fusion = np.matmul(query_feat_fusion, np.transpose(gall_feat))
                cmc_fusion, mAP_fusion, mINP_fusion = eval_sysu(-distmat_fusion, loader.query_label, loader.gall_label, loader.query_cam,
                                        loader.gall_cam)
                all_cmc_fusion += cmc_fusion
                all_mAP_fusion += mAP_fusion
                all_mINP_fusion += mINP_fusion
            if 'Text' in config.test_modality:
                distmat_text = np.matmul(query_feat_text, np.transpose(gall_feat))
                cmc_text, mAP_text, mINP_text = eval_sysu(-distmat_text, loader.query_label, loader.gall_label, loader.query_cam,
                                        loader.gall_cam)
                all_cmc_text += cmc_text
                all_mAP_text += mAP_text
                all_mINP_text += mINP_text
        
        if 'IR' in config.test_modality:
            all_cmc_ir /= 10.0
            all_mAP_ir /= 10.0
            all_mINP_ir /= 10.0
            result_dict['IR'] = (all_mINP_ir, all_mAP_ir, all_cmc_ir)
        if 'Fusion' in config.test_modality:
            all_cmc_fusion /= 10.0
            all_mAP_fusion /= 10.0
            all_mINP_fusion /= 10.0
            result_dict['Fusion'] = (all_mINP_fusion, all_mAP_fusion, all_cmc_fusion)
        if 'Text' in config.test_modality:
            all_cmc_text /= 10.0
            all_mAP_text /= 10.0
            all_mINP_text /= 10.0
            result_dict['Text'] = (all_mINP_text, all_mAP_text, all_cmc_text)


    elif loader.dataset == 'regdb':
        all_cmc_ir = 0
        all_mAP_ir = 0
        all_mINP_ir = 0
        all_cmc_fusion = 0
        all_mAP_fusion = 0
        all_mINP_fusion = 0
        all_cmc_text = 0
        all_mAP_text = 0
        all_mINP_text = 0
        for i in range(config.eval_num_regdb):
            ptr = 0
            gall_loader = loader.gallery_loaders[i]
            if 'IR' in config.test_modality and config.Fix_Visual:
                    gall_feat_for_IR = np.zeros((loader.n_gallery, embed_dim))
            if 'IR' in config.test_modality or 'Fusion' in config.test_modality or 'Text' in config.test_modality:
                gall_feat = np.zeros((loader.n_gallery, embed_dim))
            with torch.no_grad():
                for batch_idx, batch_dict in enumerate(gall_loader):
                    input = batch_dict['img']

                    batch_num = input.size(0)
                    input = Variable(input.to(device))

                    # get the feature of the last layer 
                    feat_map = base.encode_image_featmap(input,'rgb')
                    if 'IR' in config.test_modality and config.Fix_Visual:
                        feat_for_IR = base.backup_pool(feat_map).squeeze()
                        feat_for_IR = base.backup_classifier(feat_for_IR)
                        gall_feat_for_IR[ptr:ptr + batch_num, :] = feat_for_IR.detach().cpu().numpy()
                    if 'IR' in config.test_modality or 'Text' in config.test_modality or 'Fusion' in config.test_modality:
                        feat = base.base_model.visual.__getattr__(config.pooling)(feat_map).squeeze()
                        feat = base.classifier(feat)
                        gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                    else: 
                        ValueError('Error: test_modality not found!')
                    ptr = ptr + batch_num
            
            if 'IR' in config.test_modality:
                query_feat_ir = query_feat_ir_list[i]
                if config.regdb_test_mode == 't-v':
                    if config.Fix_Visual:
                        distmat_ir = np.matmul(query_feat_ir, np.transpose(gall_feat_for_IR))
                    else:
                        distmat_ir = np.matmul(query_feat_ir, np.transpose(gall_feat))
                    cmc_ir, mAP_ir, mINP_ir = eval_regdb(-distmat_ir, loader.query_label, loader.gall_label)
                else:
                    if config.Fix_Visual:
                        distmat_ir = np.matmul(gall_feat_for_IR, np.transpose(query_feat_ir))
                    else:
                        distmat_ir = np.matmul(gall_feat, np.transpose(query_feat_ir))
                    distmat_ir = np.matmul(gall_feat, np.transpose(query_feat_ir))
                    cmc_ir, mAP_ir, mINP_ir = eval_regdb(-distmat_ir, loader.gall_label, loader.query_label)
                all_cmc_ir += cmc_ir
                all_mAP_ir += mAP_ir
                all_mINP_ir += mINP_ir
            
            if 'Fusion' in config.test_modality:
                query_feat_fusion = query_feat_fusion_list[i]
                if config.regdb_test_mode == 't-v':
                    if config.CAT_EVAL:
                        distmat_ir_f = np.matmul(query_feat_ir_f, np.transpose(gall_feat))
                        distmat_text_f = np.matmul(query_feat_text_f, np.transpose(gall_feat))
                        distmat_fusion = np.matmul(query_feat_fusion, np.transpose(gall_feat)) + distmat_ir_f + distmat_text_f
                    else:
                        distmat_fusion = np.matmul(query_feat_fusion, np.transpose(gall_feat))
                    cmc_fusion, mAP_fusion, mINP_fusion = eval_regdb(-distmat_fusion, loader.query_label, loader.gall_label)
                else:
                    distmat_fusion = np.matmul(gall_feat, np.transpose(query_feat_fusion))
                    cmc_fusion, mAP_fusion, mINP_fusion = eval_regdb(-distmat_fusion, loader.gall_label, loader.query_label)
                all_cmc_fusion += cmc_fusion
                all_mAP_fusion += mAP_fusion
                all_mINP_fusion += mINP_fusion
            
            if 'Text' in config.test_modality:
                query_feat_text = query_feat_text_list[i]
                if config.regdb_test_mode == 't-v':
                    distmat_text = np.matmul(query_feat_text, np.transpose(gall_feat))
                    cmc_text, mAP_text, mINP_text = eval_regdb(-distmat_text, loader.query_label, loader.gall_label)
                else:
                    distmat_text = np.matmul(gall_feat, np.transpose(query_feat_text))
                    cmc_text, mAP_text, mINP_text = eval_regdb(-distmat_text, loader.gall_label, loader.query_label)
                all_cmc_text += cmc_text
                all_mAP_text += mAP_text
                all_mINP_text += mINP_text

        num_eval = config.eval_num_regdb
        if 'IR' in config.test_modality:
            all_cmc_ir /= num_eval
            all_mAP_ir /= num_eval
            all_mINP_ir /= num_eval
            result_dict['IR'] = (all_mINP_ir, all_mAP_ir, all_cmc_ir)
        if 'Fusion' in config.test_modality:
            all_cmc_fusion /= num_eval
            all_mAP_fusion /= num_eval
            all_mINP_fusion /= num_eval
            result_dict['Fusion'] = (all_mINP_fusion, all_mAP_fusion, all_cmc_fusion)
        if 'Text' in config.test_modality:
            all_cmc_text /= num_eval
            all_mAP_text /= num_eval
            all_mINP_text /= num_eval
            result_dict['Text'] = (all_mINP_text, all_mAP_text, all_cmc_text)
    
    elif loader.dataset == 'llcm':
        all_cmc_ir = 0
        all_mAP_ir = 0
        all_mINP_ir = 0
        all_cmc_fusion = 0
        all_mAP_fusion = 0
        all_mINP_fusion = 0
        all_cmc_text = 0
        all_mAP_text = 0
        all_mINP_text = 0
        for i in range(10):
            ptr = 0
            gall_loader = loader.gallery_loaders[i]
            if 'IR' in config.test_modality and config.Fix_Visual:
                    gall_feat_for_IR = np.zeros((loader.n_gallery, embed_dim))
            if 'IR' in config.test_modality or 'Fusion' in config.test_modality or 'Text' in config.test_modality:
                gall_feat = np.zeros((loader.n_gallery, embed_dim))
            with torch.no_grad():
                for batch_idx, batch_dict in enumerate(gall_loader):
                    input = batch_dict['img']

                    batch_num = input.size(0)
                    input = Variable(input.to(device))

                    # get the feature of the last layer 
                    feat_map = base.encode_image_featmap(input,'rgb')
                    if 'IR' in config.test_modality and config.Fix_Visual:
                        feat_for_IR = base.backup_pool(feat_map).squeeze()
                        feat_for_IR = base.backup_classifier(feat_for_IR)
                        gall_feat_for_IR[ptr:ptr + batch_num, :] = feat_for_IR.detach().cpu().numpy()
                    if 'IR' in config.test_modality or 'Text' in config.test_modality or 'Fusion' in config.test_modality:
                        feat = base.base_model.visual.__getattr__(config.pooling)(feat_map).squeeze()
                        feat = base.classifier(feat)
                        gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                    else: 
                        ValueError('Error: test_modality not found!')
                    ptr = ptr + batch_num
            
            if 'IR' in config.test_modality:
                if config.Fix_Visual:
                    distmat_ir = np.matmul(query_feat_ir, np.transpose(gall_feat_for_IR))
                else:
                    distmat_ir = np.matmul(query_feat_ir, np.transpose(gall_feat))
                cmc_ir, mAP_ir, mINP_ir = eval_sysu(-distmat_ir, loader.query_label, loader.gall_label, loader.query_cam,
                                        loader.gall_cam)
                all_cmc_ir += cmc_ir
                all_mAP_ir += mAP_ir
                all_mINP_ir += mINP_ir
            
            if 'Fusion' in config.test_modality:
                if config.CAT_EVAL:
                    distmat_ir_f = np.matmul(query_feat_ir_f, np.transpose(gall_feat))
                    distmat_text_f = np.matmul(query_feat_text_f, np.transpose(gall_feat))
                    distmat_fusion = np.matmul(query_feat_fusion, np.transpose(gall_feat)) + distmat_ir_f + distmat_text_f
                else:
                    distmat_fusion = np.matmul(query_feat_fusion, np.transpose(gall_feat))
                cmc_fusion, mAP_fusion, mINP_fusion = eval_llcm(-distmat_fusion, loader.query_label, loader.gall_label, loader.query_cam,
                                        loader.gall_cam)
                all_cmc_fusion += cmc_fusion
                all_mAP_fusion += mAP_fusion
                all_mINP_fusion += mINP_fusion
            if 'Text' in config.test_modality:
                distmat_text = np.matmul(query_feat_text, np.transpose(gall_feat))
                cmc_text, mAP_text, mINP_text = eval_llcm(-distmat_text, loader.query_label, loader.gall_label, loader.query_cam,
                                        loader.gall_cam)
                all_cmc_text += cmc_text
                all_mAP_text += mAP_text
                all_mINP_text += mINP_text
        if 'IR' in config.test_modality:
            all_cmc_ir /= 10.0
            all_mAP_ir /= 10.0
            all_mINP_ir /= 10.0
            result_dict['IR'] = (all_mINP_ir, all_mAP_ir, all_cmc_ir)
        if 'Fusion' in config.test_modality:
            all_cmc_fusion /= 10.0
            all_mAP_fusion /= 10.0
            all_mINP_fusion /= 10.0
            result_dict['Fusion'] = (all_mINP_fusion, all_mAP_fusion, all_cmc_fusion)
        if 'Text' in config.test_modality:
            all_cmc_text /= 10.0
            all_mAP_text /= 10.0
            all_mINP_text /= 10.0
            result_dict['Text'] = (all_mINP_text, all_mAP_text, all_cmc_text)

    return result_dict