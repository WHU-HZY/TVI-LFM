import numpy as np
from PIL import Image
import pdb
import os
import json

data_path = "/data1/hzy_data/SYSU-MM01/"

rgb_cameras = ['cam1','cam2','cam4','cam5']
ir_cameras = ['cam3','cam6']

# load text info
with open('/data1/hzy_data/SYSU-MM01/output_full_aug.json', 'r') as f:
    text_dict = json.load(f)

# load id info
file_path_train = os.path.join(data_path,'exp/train_id.txt')
file_path_val   = os.path.join(data_path,'exp/val_id.txt')
with open(file_path_train, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_train = ["%04d" % x for x in ids]
    
with open(file_path_val, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_val = ["%04d" % x for x in ids]
    
# combine train and val split   
id_train.extend(id_val) 

files_rgb = []
files_ir = []
files_text = [] # 元素：对于每一张RGB图片的描述
files_text_id = [] # 元素：对于每一张RGB图片的描述对应的id
files_text_llm_aug = [] # 元素：对于每一张RGB图片的llm增强描述列表
files_text_llm_ea_aug = [] # 元素：对于每一张RGB图片的llm+ea增强描述列表
files_text_eaaug = [] # 元素：对于每一张RGB图片的ea增强描述列表

for id in sorted(id_train):
    for cam in rgb_cameras:
        # 得到每一个图片的path
        img_dir = os.path.join(data_path,cam,id)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
            print(new_files)
            # files_rgb.extend(new_files)
            # files_text.extend([text_dict[i]["description"] for i in new_files])
            # files_text_id.extend([str(int(id)) for i in new_files])
            # files_text_llm_aug.extend([text_dict[i]["llm_rephrase"] for i in new_files])
            files_text_eaaug.extend([text_dict[i]["ori_aug_description"] for i in new_files])
    # for cam in ir_cameras:
    #     img_dir = os.path.join(data_path,cam,id)
    #     if os.path.isdir(img_dir):
    #         new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
    #         files_ir.extend(new_files)

# # relabel
# pid_container = set()
# for img_path in files_ir:
#     pid = int(img_path[-13:-9])
#     pid_container.add(pid)
# pid2label = {pid:label for label, pid in enumerate(pid_container)}
# files_text_id = [pid2label[int(i)] for i in files_text_id]

# fix_image_width = 144
# fix_image_height = 288
# def read_imgs(train_image):
#     train_img = []
#     train_label = []
#     for img_path in train_image:
#         # img
#         img = Image.open(img_path)
#         img = img.resize((fix_image_width, fix_image_height), Image.ANTIALIAS)
#         pix_array = np.array(img)

#         train_img.append(pix_array) 
        
#         # label
#         pid = int(img_path[-13:-9])
#         pid = pid2label[pid]
#         train_label.append(pid)
#     return np.array(train_img), np.array(train_label)
       
# text
# np.save(data_path + 'train_text_id.npy', np.array(files_text_id))
# np.save(data_path + 'train_text.npy', files_text)
# print(len(files_text_llm_aug))
# print(len(files_text_llm_ea_aug))
# np.save(data_path + 'train_text_llm_aug.npy', np.array(files_text_llm_aug))
np.save(data_path + 'train_text_eaaug_list.npy', np.array(files_text_eaaug))
# # rgb imges
# train_img, train_label = read_imgs(files_rgb)
# np.save(data_path + 'train_rgb_resized_img.npy', train_img)
# np.save(data_path + 'train_rgb_resized_label.npy', train_label)

# # ir imges
# train_img, train_label = read_imgs(files_ir)
# np.save(data_path + 'train_ir_resized_img.npy', train_img)
# np.save(data_path + 'train_ir_resized_label.npy', train_label)
