import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
import math
from einops import rearrange

import math

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def extract_patches(img, patch_size=512, stride=512):
    _, _, h, w = img.size()
    num_h = (h - patch_size + stride - 1) // stride + 1
    num_w = (w - patch_size + stride - 1) // stride + 1
    patches = []
    for i in range(num_h):
        for j in range(num_w):
            y_start = min(i * stride, h - patch_size)
            x_start = min(j * stride, w - patch_size)
            patch = img[:, :, y_start:y_start + patch_size, x_start:x_start + patch_size]
            patches.append(patch)
    patches = torch.cat(patches, dim=0)
    return patches, num_h, num_w


def reassemble_patches(patches, num_h, num_w, h, w, patch_size=512, stride=256):
    result = torch.zeros(1, patches.size(1), h, w).to(patches.device)
    norm_map = torch.zeros(1, 1, h, w).to(patches.device)
    patches = F.interpolate(patches, scale_factor=8, mode='bilinear') / 64
    patch_idx = 0
    for i in range(num_h):
        for j in range(num_w):
            y_start = min(i * stride, h - patch_size)
            x_start = min(j * stride, w - patch_size)
            result[:, :, y_start:y_start + patch_size, x_start:x_start + patch_size] += patches[patch_idx]
            norm_map[:, :, y_start:y_start + patch_size, x_start:x_start + patch_size] += 1
            patch_idx += 1

    result /= norm_map
    return result


def visualize_density_map(img, density_map, prompt, gt_count, pred_count, save_path):
    """
    Visualize source image, density map and text information
    """
    # Convert tensor image to numpy for visualization
    if torch.is_tensor(img):
        img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    else:
        img_np = img
    
    # Process density map
    if torch.is_tensor(density_map):
        density_map = density_map.detach().cpu().numpy()
    
    # Resize density map to match image size (48x48 -> 384x384)
    # density_map_resized = cv2.resize(density_map, (img_np.shape[1], img_np.shape[0]), 
    #                                 interpolation=cv2.INTER_LINEAR)
    density_map_resized = cv2.resize(density_map, (img_np.shape[1], img_np.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)

    # mask = density_map_resized > 1e-3 * 60
    # density_map_resized = density_map_resized * mask
    
    # Apply colormap to density map (jet colormap like in visual_rec8k.py)
    density_colored = plt.cm.jet(density_map_resized / (density_map_resized.max() + 1e-8))[:, :, :3]
    
    # Blend image and density map with transparency
    alpha = 0.6
    blended = (1 - alpha) * img_np + alpha * density_colored
    blended = np.clip(blended, 0, 1)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot source image
    ax1.imshow(img_np)
    ax1.set_title('Source Image')
    ax1.axis('off')
    
    # Plot blended image with density map
    ax2.imshow(blended)
    ax2.set_title('Image with Density Map Overlay')
    ax2.axis('off')

    # Calculate absolute error
    abs_error = abs(gt_count - pred_count)
    
    # Add text information
    text_str = f"Prompt: {prompt}\nGT Count: {gt_count:.1f}\nPred Count: {pred_count:.1f}\nAbs Error: {abs_error:.1f}"
    plt.figtext(0.5, 0.01, text_str, ha='center', fontsize=12, 
                bbox=dict(facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print("processed: ", save_path)

def process_attn(avg_attn, base_size=384):
    """
    处理注意力图，分别处理普通和latent的注意力
    """
    size12 = base_size // 32
    size24 = base_size // 16
    size48 = base_size // 8
    
    # 普通注意力
    attn_dict = {size12: [], size24: [], size48: []}
    # latent注意力
    attn_latent_dict = {size12: [], size24: [], size48: []}
    
    # 处理普通注意力
    for k in avg_attn:
        if not k.endswith('_latent'):
            for attn in avg_attn[k]:
                if torch.is_tensor(attn):
                    seq_len = attn.shape[1]
                else:
                    seq_len = attn.shape[1]
                
                size = int(math.sqrt(seq_len))
                if size in attn_dict:
                    attn_reshaped = rearrange(attn, 'b (h w) c -> b c h w', h=size)
                    attn_dict[size].append(attn_reshaped)
    
    # 处理latent注意力
    for k in avg_attn:
        if k.endswith('_latent'):
            for attn in avg_attn[k]:
                if torch.is_tensor(attn):
                    seq_len = attn.shape[1]
                else:
                    seq_len = attn.shape[1]
                
                size = int(math.sqrt(seq_len))
                if size in attn_latent_dict:
                    attn_reshaped = rearrange(attn, 'b (h w) c -> b c h w', h=size)
                    attn_latent_dict[size].append(attn_reshaped)
    
    # 计算平均值
    def compute_avg(attn_list_dict):
        result = {}
        for size, attn_list in attn_list_dict.items():
            if len(attn_list) > 0:
                # 在dim=0上堆叠并在dim=0上求平均
                stacked = torch.stack(attn_list)  # [n, b, c, h, w]
                avg_attn = stacked.mean(0)  # [b, c, h, w]
                # 在channel维度(dim=1)上求平均，得到 [b, h, w]
                avg_attn = avg_attn.mean(1, keepdim=True)  # [b, 1, h, w]
                result[size] = avg_attn
            else:
                result[size] = None
        return result
    
    attn_avg = compute_avg(attn_dict)
    attn_latent_avg = compute_avg(attn_latent_dict)
    
    # 返回结果
    return {
        'attn12': attn_avg.get(size12),
        'attn24': attn_avg.get(size24), 
        'attn48': attn_avg.get(size48),
        'attn12_latent': attn_latent_avg.get(size12),
        'attn24_latent': attn_latent_avg.get(size24),
        'attn48_latent': attn_latent_avg.get(size48)
    }

def visualize_attn_map(avg_attn, save_path, base_size=384, dpi=150):
    """
    可视化注意力图并保存
    
    Args:
        avg_attn: 从attention_store获取的平均注意力字典
        save_path: 保存目录路径
        base_size: 基础尺寸，默认为512
        dpi: 图像分辨率
    """
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 处理注意力图
    processed_attn = process_attn(avg_attn, base_size)
    
    # 可视化每个注意力图
    for attn_name, attn_tensor in processed_attn.items():
        if attn_tensor is None:
            print(f"跳过 {attn_name}，没有可用的数据")
            continue
            
        # 确保是PyTorch tensor
        if torch.is_tensor(attn_tensor):
            # 取第一个batch，维度为 [1, h, w]
            attn_np = attn_tensor[0, 0].detach().cpu().numpy()  # [h, w]
        else:
            attn_np = attn_tensor[0, 0]  # [h, w]
        
        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # 绘制热力图
        im = ax.imshow(attn_np, cmap='viridis', aspect='equal')
        
        # 设置标题和颜色条
        ax.set_title(f'Attention Map: {attn_name}\nShape: {attn_np.shape}', fontsize=12, pad=20)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight', rotation=270, labelpad=15)
        
        # 添加网格线（可选）
        ax.grid(False)
        
        # 保存图像
        filename = os.path.join(save_path, f'{attn_name}.png')
        plt.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"已保存: {filename}")
