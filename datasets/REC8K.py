import torch
from torch.utils.data import Dataset
from glob import glob
import os
import numpy as np
import h5py
import torchvision.transforms.functional as F
import random
from PIL import Image
from torchvision import transforms
import cv2
import json
from transformers import CLIPTokenizer


def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = 0 if res_h < 0 else random.randint(0, res_h)
    j = 0 if res_w < 0 else random.randint(0, res_w)

    h = 0 if res_h < 0 else random.randint(0, im_h)
    w = 0 if res_w < 0 else random.randint(0, im_w)

    # i = random.randint(0, res_h)
    # j = random.randint(0, res_w)
    return i, j, h, w


class ObjectCount(Dataset):
    def __init__(self, root, crop_size, downsample_ratio, method='train', concat_size=224):
        super(ObjectCount, self).__init__()
        #self.im_list = sorted(glob(os.path.join(root, 'images/*.jpg')))
        assert crop_size % downsample_ratio == 0
        assert method in ['train', 'val', 'test'], f"Invalid method: {method}. Must be 'train', 'val', or 'test'."
        self.root_path = root
        self.crop_size = crop_size
        self.down_ratio = downsample_ratio
        self.method = method
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # self.tokenizer = CLIPTokenizer.from_pretrained("/home/ubuntu/jyt/checkpoints/clip")
        self.concat_size = concat_size

        with open(os.path.join(root, 'splits_add_prompt_length.json'), 'r') as f:
            data_split = json.load(f)[method]
            self.im_list = [[x[0], x[1], x[2]] for x in data_split]

        with open(os.path.join(root, 'annotations.json'), 'r') as f:
            self.annotations = json.load(f)

        self.cls_dict = {}
        with open(os.path.join(root, 'ImageClasses_REC8K.txt'), "r", encoding="utf-8") as f:
            for line in f:
                self.cls_dict[line.strip().split()[0]] = line.strip().split()[1]

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        im_name = self.im_list[item][0]
        im_path = os.path.join(self.root_path, 'rec-8k', im_name)
        prompt = self.im_list[item][1]

        den_path = os.path.join(self.root_path, f'gt_density_map_{self.crop_size}', im_name[:-4], prompt.lower().strip().replace(" ", "_") + ".npy")
        # den_path = os.path.join(self.root_path, f'gt_density_map_384', im_name[:-4], prompt.lower().strip().replace(" ", "_") + ".npy")

        # den_all_path = os.path.join(self.root_path, f'gt_density_map_{self.crop_size}', im_name[:-4], im_name[:-4] + ".npy")

        img = Image.open(im_path).convert('RGB')
        wd, ht = img.size
        re_w = self.crop_size if wd < self.crop_size else wd
        re_h = self.crop_size if ht < self.crop_size else ht
        img = img.resize((re_w, re_h), Image.Resampling.BICUBIC)
        cls_name = self.cls_dict[os.path.basename(im_path)]
        pts = self.annotations[os.path.basename(im_path)][prompt]['points']
        # attribute = self.annotations[os.path.basename(im_path)][prompt]['attribute']

        prompt_attn_mask = torch.zeros(77)
        # cls_name_tokens = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
        # cls_name_length = cls_name_tokens['input_ids'].shape[1]
        cls_name_length = self.im_list[item][2]
        prompt_attn_mask[1: 1 + cls_name_length] = 1

        # cls_name_position_mask = self.get_cls_name_position_mask(prompt, cls_name)

        if self.method == 'train':
            if random.random() > 0.5:
                out_img = img
                wd, ht = img.size
                den_map = np.load(den_path)
                # den_all_map = np.load(den_all_path)
                img_attn_map = np.ones((ht, wd))

            else:
                if random.random() > 0.5:
                    rand_img = random.sample(self.im_list, 1)[0]

                    rand_cls = self.cls_dict[os.path.basename(rand_img[0])]

                    out_img = img
                    if rand_cls != cls_name:
                        wd, ht = img.size
                        den_map = np.zeros((self.crop_size, self.crop_size))
                        # den_all_map = np.zeros((self.crop_size, self.crop_size))
                        # den_map = np.zeros((384, 384))
                        prompt = rand_cls
                        img_attn_map = np.zeros((ht, wd))
                        prompt_attn_mask = torch.zeros(77)
                        # cls_name_tokens = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
                        # cls_name_length = cls_name_tokens['input_ids'].shape[1]
                        cls_name_length = self.im_list[item][2]
                        prompt_attn_mask[1: 1 + cls_name_length] = 1
                    else:
                        wd, ht = img.size
                        den_map = np.load(den_path)
                        # den_all_map = np.load(den_all_path)
                        img_attn_map = np.ones((ht, wd))
                else:
                    out_img = img
                    wd, ht = img.size
                    den_map = np.load(den_path)
                    # den_all_map = np.load(den_all_path)
                    img_attn_map = np.ones((ht, wd))
              
            img, den_map, img_attn_map = self.train_transform_density(out_img, den_map, img_attn_map)
            return img, den_map, prompt, prompt_attn_mask, img_attn_map
        else:
            img = img.resize((self.crop_size, self.crop_size), Image.Resampling.BICUBIC)
            return self.transform(img), len(pts), prompt, prompt_attn_mask, os.path.basename(im_path).split('.')[0]

    def train_transform_density(self, img, den_map, img_attention_map):
        
        img_attention_map = cv2.resize(img_attention_map, (int(self.crop_size / 8), int(self.crop_size / 8)), interpolation=cv2.INTER_NEAREST)
        img = img.resize((self.crop_size, self.crop_size), Image.Resampling.BICUBIC)

        # den_map = den_map.reshape([self.crop_size // 8, 8, self.crop_size // 8, 8]).sum(
        #     axis=(1, 3))

        if random.random() > 0.5:
            img = F.hflip(img)
            den_map = np.fliplr(den_map)
            img_attention_map = np.fliplr(img_attention_map)

        return self.transform(img), torch.from_numpy(den_map.copy()).float().unsqueeze(0), torch.from_numpy(img_attention_map.copy()).float().unsqueeze(0)

    def get_cls_name_position_mask(self, prompt, cls_name):
        """
        获取cls_name在完整prompt编码中的位置掩码
        返回形状为(77,)的tensor，cls_name对应的token位置为1，其他为0
        """
        # 编码完整prompt
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=True, return_tensors='pt')
        prompt_ids = prompt_tokens['input_ids'][0]  # 形状: [seq_len]
        
        # 编码单独的cls_name（不添加特殊token）
        cls_tokens = self.tokenizer(cls_name, add_special_tokens=False, return_tensors='pt')
        cls_ids = cls_tokens['input_ids'][0]  # 形状: [cls_len]
        
        # 创建位置掩码
        position_mask = torch.zeros(77)
        
        # 在prompt中查找cls_name的token序列
        prompt_list = prompt_ids.tolist()
        cls_list = cls_ids.tolist()
        cls_len = len(cls_list)
        
        # 在prompt token序列中搜索cls_name的token序列
        for i in range(len(prompt_list) - cls_len + 1):
            if prompt_list[i:i+cls_len] == cls_list:
                # 找到匹配，设置对应位置为1
                start_pos = i
                end_pos = i + cls_len
                position_mask[start_pos:end_pos] = 1
                break
        
        return position_mask
