from torch.utils.data import DataLoader, default_collate, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch
import logging
from functools import partial
from utils.helper import SaveHandler, AverageMeter
from utils.trainer import Trainer
from models.GDecoder_D import GDecoder
# from datasets.REC8K import ObjectCount
from datasets.FSC147 import ObjectCount
import numpy as np
import os
import time
import random
import torch.nn.functional as F
import torch.nn as nn
from utils.losses import cal_avg_ms_ssim, get_reg_loss
from utils.tools import extract_patches, reassemble_patches, visualize_density_map, visualize_attn_map, adjust_learning_rate
import sys
import math
from torch import distributed as dist

from PIL import Image
from torchvision import transforms
import cv2
from transformers import CLIPTokenizer
import json


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    den = torch.stack(transposed_batch[1], 0)  # the number of points is not fixed, keep it as a list of tensor
    prompt = transposed_batch[2]
    return images, den, prompt


def train_setup(rank, world_size, args):
    trainer = Reg_Trainer(args)
    trainer.setup(rank, world_size)
    trainer.train()

class Reg_Trainer(Trainer):
    def setup(self, rank=0, world_size=1):
        self.rank = rank
        args = self.args
        if args.seed != -1:
            setup_seed(args.seed)
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{rank}')
            torch.cuda.set_device(torch.cuda.current_device())
        else:
            raise Exception('GPU is not available')

        self.d_ratio = args.downsample_ratio

        self.datasets = {x: ObjectCount(args.data_dir,
                                        crop_size=args.crop_size,
                                        downsample_ratio=self.d_ratio,
                                        method=x,
                                        concat_size=args.concat_size) for x in ['train', 'val', 'test']}

        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          batch_size=(args.batch_size // world_size
                                                      if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                        drop_last=(True if x == 'train' else False),
                                          collate_fn=(train_collate if x=='train' else default_collate),
                                          num_workers=args.num_workers,
                                          pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'val', 'test']}


        self.model = GDecoder(
            patch_size=16, embed_dim=768, depth=12, num_heads=12,
            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        
        self.model.to(self.device)

        backbone_params = dict()
        freeze_params = dict()
        non_backbone_params = dict()
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith('blocks'):
                print("low lr: ", n)
                backbone_params[n] = p
            elif n.startswith('bert') or n.startswith('tokenizer'):
                freeze_params[n] = p
            else:
                non_backbone_params[n] = p

        self.optimizer = torch.optim.AdamW(
            [
                {'params': non_backbone_params.values()},
                {'params': backbone_params.values()},
                {'params': freeze_params.values(), 'lr': 0.0}
            ],
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95)
        )

        # self.optimizer = torch.optim.AdamW([
        #     {'params': self.model.unet.parameters(),
        #      'lr': args.lr,
        #      'weight_decay': args.weight_decay}], betas=(0.9, 0.95))

        self.start_epoch = 0
        self.epoch = 0

        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu')

            if 'pos_embed' in checkpoint['model'] and checkpoint['model']['pos_embed'].shape != self.model.state_dict()['pos_embed'].shape:
                print(f"Removing key pos_embed from pretrained checkpoint")
                del checkpoint['model']['pos_embed']
        
            if 'decoder_pos_embed' in checkpoint['model'] and checkpoint['model']['decoder_pos_embed'].shape != self.model.state_dict()['decoder_pos_embed'].shape:
                print(f"Removing key decoder_pos_embed from pretrained checkpoint")
                del checkpoint['model']['decoder_pos_embed']
            self.model.load_state_dict(checkpoint['model'], strict=False)

        # if args.eval:
        #     self.model.load_state_dict(torch.load(args.eval, self.device), strict=False)
            # self.model.to(rank)

        self.best_mae = np.inf
        self.best_mse = np.inf
        self.save_list = SaveHandler(num=args.max_num)

        self.train()

        self.absolute_errors = {"test":{}, "val":{}}

    def train(self):
        args = self.args
        for epoch in range(self.start_epoch, args.epochs):
            if self.rank == 0:
                logging.info('-' * 50 + "Epoch:{}/{}".format(epoch, args.epochs - 1) + '-' * 50)
            self.epoch = epoch
            self.train_epoch()
            if self.epoch >= args.start_val and self.epoch % self.args.val_epoch == 0 and self.rank == 0:
                self.val_epoch()

    def train_epoch(self):
        epoch_reg_loss = AverageMeter()
        epoch_sem_loss = AverageMeter()
        epoch_RRC2_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()

        for step, (inputs, den_map, caption) in enumerate(
                self.dataloaders['train']):

            adjust_learning_rate(self.optimizer, step / len(self.dataloaders['train']) + self.epoch, self.args)
            # self.dataloaders['train'].sampler.set_epoch(step)
            inputs = inputs.to(self.device).half()
            gt_den_maps = den_map.to(self.device).half() * self.args.scale
            self.model.train()
            with torch.cuda.amp.autocast():
                N = inputs.shape[0]
                pred_den = self.model([inputs, caption])

                reg_loss = get_reg_loss(pred_den, gt_den_maps, threshold=1e-3 * self.args.scale)

                epoch_reg_loss.update(reg_loss.item(), N)

                loss = reg_loss

                if not math.isfinite(loss.item()):
                    logging.info("Loss is {}, stopping training".format(loss))
                    sys.exit(1)

                gt_counts = torch.sum(gt_den_maps.view(N, -1), dim=1).detach().cpu().numpy() / self.args.scale
                pred_counts = torch.sum(pred_den.view(N, -1), dim=1).detach().cpu().numpy() / self.args.scale
                diff = pred_counts - gt_counts
                epoch_mae.update(np.mean(np.abs(diff)).item(), N)
                epoch_mse.update(np.mean(diff * diff), N)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.rank == 0:
            logging.info(
                'Epoch {} Train, reg:{:.4f}, mae:{:.2f}, mse:{:.2f}, Cost: {:.1f} sec '
                .format(self.epoch, epoch_reg_loss.getAvg(), epoch_mae.getAvg(),
                        np.sqrt(epoch_mse.getAvg()), (time.time() - epoch_start)))

            # if self.epoch % 20 == 0:
            #     model_state_dict = self.model.state_dict()
            #     save_path = os.path.join(self.save_dir, "resume_ckpt.tar")
            #     torch.save({
            #         'epoch': self.epoch,
            #         'optimizer_state_dict': self.optimizer.state_dict(),
            #         'model_state_dict': model_state_dict,
            #     }, save_path)
            #     self.save_list.append(save_path)

    def val_epoch(self, visual=False, analysis=False):
        epoch_start = time.time()
        self.model.set_eval()
        epoch_res = []

        see_mae = 0
        see_rmse = 0
        unsee_mae = 0
        unsee_rmse = 0
        see_counter = 0
        unsee_counter = 0
        train_exp = self.get_train_exp()
        # print("train_exp: ", train_exp[0:10])

        for inputs, gt_counts, captions, name in self.dataloaders['val']:
            inputs = inputs.to(self.device)
            cropped_imgs, num_h, num_w = extract_patches(inputs, patch_size=self.args.crop_size,
                                                         stride=self.args.stride)
            outputs = []
            with torch.set_grad_enabled(False):
                # num_chunks = (cropped_imgs.size(0) + self.args.batch_size - 1) // self.args.batch_size
                # for i in range(num_chunks):
                #     start_idx = i * self.args.batch_size
                #     end_idx = min((i + 1) * self.args.batch_size, cropped_imgs.size(0))
                #     outputs_partial, avg_attn = self.model(cropped_imgs[start_idx:end_idx], captions * (end_idx - start_idx), gt_attn_mask.repeat((end_idx - start_idx), 1, 1, 1))
                #     outputs.append(outputs_partial)
                # results = reassemble_patches(torch.cat(outputs, dim=0), num_h, num_w, inputs.size(2), inputs.size(3),
                #                              patch_size=self.args.crop_size, stride=self.args.stride)

                results = self.model([inputs, captions])
                res = gt_counts[0].item() - torch.sum(results).item() / self.args.scale
                epoch_res.append(res)

                if name[0]+".jpg" not in absolute_errors[split].keys():
                    self.absolute_errors[split][name[0]+".jpg"] = {}
                self.absolute_errors[split][name[0]+".jpg"][captions[0]] = pred_cnt - gt_cnt
            if analysis:
                if captions[0].lower().strip() in train_exp:
                    see_mae += abs(res)
                    see_rmse += abs(res) ** 2
                    see_counter += 1
                else:
                    unsee_mae += abs(res)
                    unsee_rmse += abs(res) ** 2
                    unsee_counter += 1
            
            if visual:
                results = results.squeeze(0).squeeze(0)
                pred_count = torch.sum(results).item() / self.args.scale
                gt_count = gt_counts[0].item()
                prompt_text = captions[0] if isinstance(captions[0], str) else "Unknown prompt"

                save_dir = os.path.join(self.save_dir, "val", name[0])
                os.makedirs(save_dir, exist_ok=True)

                attn_save_dir = os.path.join(self.save_dir, "val", name[0], prompt_text.replace(" ", "_"))
                os.makedirs(attn_save_dir, exist_ok=True)

                visualize_density_map(inputs, results.detach().cpu(), prompt_text, gt_count, pred_count, os.path.join(save_dir, prompt_text.replace(" ", "_") + ".png"))
                
                # visualize_attn_map(avg_attn, gt_attn_mask, attn_save_dir)

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        logging.info('Epoch {} Val, MAE: {:.2f}, MSE: {:.2f} Cost {:.1f} sec'
                            .format(self.epoch, mae, mse, (time.time() - epoch_start)))

        if analysis:
            see_mae = see_mae / see_counter
            see_rmse = (see_rmse / see_counter) ** 0.5
            unsee_mae = unsee_mae / unsee_counter
            unsee_rmse = (unsee_rmse / unsee_counter) ** 0.5

            logging.info('Epoch {} Val, SeeMAE: {:.2f}, SeeMSE: {:.2f}, UnseeMAE: {:.2f}, UnseeMSE: {:.2f}'
                            .format(self.epoch, see_mae, see_rmse, unsee_mae, unsee_rmse))

        model_state_dict = self.model.state_dict()

        if mae < self.best_mae:
            self.best_mae = mae
            self.best_mse = mse
            if visual is False and analysis is False:
                torch.save(model_state_dict, os.path.join(self.save_dir, 'best_model.pth'))
                logging.info("Save best model: MAE: {:.2f} MSE:{:.2f} model epoch {}".format(mae, mse, self.epoch))
                self.test_epoch()
                print("Best Result: MAE: {:.2f} MSE:{:.2f}".format(self.best_mae, self.best_mse))

    def test_epoch(self, visual=False, analysis=False):
        epoch_start = time.time()
        self.model.set_eval()
        epoch_res = []
        see_mae = 0
        see_rmse = 0
        unsee_mae = 0
        unsee_rmse = 0
        see_counter = 0
        unsee_counter = 0
        train_exp = self.get_train_exp()

        for inputs, gt_counts, captions, prompt_attn_mask, name in self.dataloaders['test']:
            inputs = inputs.to(self.device)
            gt_attn_mask = prompt_attn_mask.to(self.device).unsqueeze(2).unsqueeze(3)
            cropped_imgs, num_h, num_w = extract_patches(inputs, patch_size=self.args.crop_size,
                                                         stride=self.args.stride)
            outputs = []
            with torch.set_grad_enabled(False):
                # num_chunks = (cropped_imgs.size(0) + self.args.batch_size - 1) // self.args.batch_size
                # for i in range(num_chunks):
                #     start_idx = i * self.args.batch_size
                #     end_idx = min((i + 1) * self.args.batch_size, cropped_imgs.size(0))
                #     outputs_partial, avg_attn = self.model(cropped_imgs[start_idx:end_idx], captions * (end_idx - start_idx), gt_attn_mask.repeat((end_idx - start_idx), 1, 1, 1))
                #     outputs.append(outputs_partial)
                # results = reassemble_patches(torch.cat(outputs, dim=0), num_h, num_w, inputs.size(2), inputs.size(3),
                #                              patch_size=self.args.crop_size, stride=self.args.stride)
                results, _ = self.model(inputs, captions, gt_attn_mask)
                res = gt_counts[0].item() - torch.sum(results).item() / self.args.scale
                epoch_res.append(res)

            if analysis:
                if captions[0].lower().strip() in train_exp:
                    see_mae += abs(res)
                    see_rmse += abs(res) ** 2
                    see_counter += 1
                else:
                    unsee_mae += abs(res)
                    unsee_rmse += abs(res) ** 2
                    unsee_counter += 1
            if visual:
                results = results.squeeze(0).squeeze(0)
                pred_count = torch.sum(results).item() / self.args.scale
                gt_count = gt_counts[0].item()
                prompt_text = captions[0] if isinstance(captions[0], str) else "Unknown prompt"
                save_dir = os.path.join(self.save_dir, "test", name[0])
                os.makedirs(save_dir, exist_ok=True)

                attn_save_dir = os.path.join(self.save_dir, "test", name[0], prompt_text.replace(" ", "_"))
                os.makedirs(attn_save_dir, exist_ok=True)

                visualize_density_map(inputs, results.detach().cpu(), prompt_text, gt_count, pred_count, os.path.join(save_dir, prompt_text.replace(" ", "_") + str(gt_counts[0].item()) + "_" + str(res) + "_.png"))
                
                # visualize_attn_map(avg_attn, gt_attn_mask, attn_save_dir)

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        logging.info('Epoch {} Test, MAE: {:.2f}, MSE: {:.2f} Cost {:.1f} sec'
                        .format(self.epoch, mae, mse, (time.time() - epoch_start)))

        if analysis:
            see_mae = see_mae / see_counter
            see_rmse = (see_rmse / see_counter) ** 0.5
            unsee_mae = unsee_mae / unsee_counter
            unsee_rmse = (unsee_rmse / unsee_counter) ** 0.5

            logging.info('Epoch {} Test, SeeMAE: {:.2f}, SeeMSE: {:.2f}, UnseeMAE: {:.2f}, UnseeMSE: {:.2f}'
                            .format(self.epoch, see_mae, see_rmse, unsee_mae, unsee_rmse))

    def visual(self):
        # print("visual val...")
        # self.val_epoch(visual=True)
        print("visual test...")
        self.test_epoch(visual=True)

    def visual_one(self, image_path, prompt):
        self.model.set_eval()
        inputs, captions, prompt_attn_mask = self.prepare_data(image_path, prompt)
        inputs = inputs.to(self.device)
        gt_attn_mask = prompt_attn_mask.to(self.device).unsqueeze(2).unsqueeze(3)
        cropped_imgs, num_h, num_w = extract_patches(inputs, patch_size=self.args.crop_size,
                                                        stride=self.args.stride)
        outputs = []
        with torch.set_grad_enabled(False):
            num_chunks = (cropped_imgs.size(0) + self.args.batch_size - 1) // self.args.batch_size
            for i in range(num_chunks):
                start_idx = i * self.args.batch_size
                end_idx = min((i + 1) * self.args.batch_size, cropped_imgs.size(0))
                outputs_partial = self.model(cropped_imgs[start_idx:end_idx], captions * (end_idx - start_idx), gt_attn_mask.repeat((end_idx - start_idx), 1, 1, 1))[0]
                outputs.append(outputs_partial)
            results = reassemble_patches(torch.cat(outputs, dim=0), num_h, num_w, inputs.size(2), inputs.size(3),
                                            patch_size=self.args.crop_size, stride=self.args.stride)

        results = results.squeeze(0).squeeze(0)
        pred_count = torch.sum(results).item() / self.args.scale

        prompt_text = captions[0] if isinstance(captions[0], str) else "Unknown prompt"

        visualize_density_map(inputs, results.detach().cpu(), prompt_text, pred_count, pred_count, os.path.join(self.save_dir, prompt_text.replace(" ", "_") + ".png"))



    def prepare_data(self, im_path, prompt):
        img = Image.open(im_path).convert('RGB')
        wd, ht = img.size
        re_w = self.args.crop_size if wd < self.args.crop_size else wd
        re_h = self.args.crop_size if ht < self.args.crop_size else ht
        img = img.resize((re_w, re_h), Image.Resampling.BICUBIC)

        prompt_attn_mask = torch.zeros(77).to(self.device)
        tokenizer = CLIPTokenizer.from_pretrained("/home/ubuntu/jyt/checkpoints/clip")
        cls_name_tokens = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
        cls_name_length = cls_name_tokens['input_ids'].shape[1]
        prompt_attn_mask[1: 1 + cls_name_length] = 1

        img = img.resize((self.args.crop_size, self.args.crop_size), Image.Resampling.BICUBIC)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return transform(img).unsqueeze(0), [prompt], prompt_attn_mask.unsqueeze(0)

    def get_train_exp(self):
        # 加载JSON文件
        with open("/home/ubuntu/jyt/datasets/REC-8K/splits_add_prompt_length.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取训练集的文本描述（使用集合去重）
        train_descriptions = set()
        for item in data.get('train', []):
            if len(item) >= 2:
                train_descriptions.add(item[1].lower().strip())
        return train_descriptions
