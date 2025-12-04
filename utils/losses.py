from torch.autograd import Variable
import torch.nn as nn
import torch
from math import exp
import torch.nn.functional as F


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2) ** 2 / float(2*sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True, dilation=1):
    kernel_size = window_size + (dilation - 1) * (window_size - 1) - 1
    mu1 = F.conv2d(img1, window, padding=kernel_size//2, dilation=dilation, groups=channel)
    mu2 = F.conv2d(img2, window, padding=kernel_size//2, dilation=dilation, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=kernel_size//2, dilation=dilation, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=kernel_size//2, dilation=dilation, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1* img2, window, padding=kernel_size//2, dilation=dilation, groups=channel) - mu1_mu2

    C1 = (0.01*1) ** 2
    C2 = (0.03*1) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def cal_avg_ms_ssim(img1, img2, level, weights=[1], window_size=5):
    if len(img1.size()) != 4:
        channel = 1
    else:
        (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if len(weights) != level:
        weights = [1] * level
    weights = torch.FloatTensor(weights)
    weights = weights / weights.sum()
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
        weights = weights.cuda(img1.get_device())

    for i in range(level):
        ssim_value = _ssim(img1, img2, window, window_size, channel, True, 1)
        if i == 0:
            avg_loss = weights[i] * (1.0 - ssim_value)
        else:
            avg_loss += weights[i] * (1.0 - ssim_value)
        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    return avg_loss

def get_normalized_map(density_map):
    B, C, H, W = density_map.size()
    mu_sum = density_map.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
    mu_normed = density_map / (mu_sum + 1e-6)
    return mu_normed


def get_reg_loss(pred, gt, threshold, level=3, window_size=3):
    mask = gt > threshold
    loss_ssim = cal_avg_ms_ssim(pred * mask, gt * mask, level=level,
                                window_size=window_size)
    mu_normed = get_normalized_map(pred)
    gt_mu_normed = get_normalized_map(gt)
    tv_loss = (nn.L1Loss(reduction='none')(mu_normed, gt_mu_normed).sum(1).sum(1).sum(1)).mean(0)
    return loss_ssim + 0.1 * tv_loss


def RRC_loss(simi, ambiguous_negative_map, positive_map):
    pos = (1 - simi) * positive_map
    neg = torch.clamp(simi, min=0) * (ambiguous_negative_map == 0) * (positive_map == 0)

    pos_num = positive_map.flatten(1).sum(dim=1)
    neg_num = ((ambiguous_negative_map == 0) * (positive_map == 0)).flatten(1).sum(dim=1)
    loss = 2 * pos.flatten(1).sum(dim=1) / (pos_num + 1e-7) + neg.flatten(1).sum(dim=1) / (neg_num + 1e-7)
    return loss.mean()

def normalize_map(m, epsilon=1e-6):
    """Normalize map to probability distribution."""
    return F.softmax(m.view(m.size(0), -1), dim=1).view_as(m) + epsilon

def get_soft_mask(density_map, threshold=1e-3*60, dilation_kernel_size=5):
    """Create softened, dilated mask from density map."""
    mask = density_map > threshold
    kernel = torch.ones(1, 1, dilation_kernel_size, dilation_kernel_size).to(density_map.device)
    soft_mask = F.conv2d(density_map.float() / (density_map.max() + 1e-6), kernel, padding=dilation_kernel_size//2)
    return normalize_map(soft_mask)  # Normalize for divergence

def semantic_attention_loss(pos_attn, neg_attn, gt_pos, gt_all, threshold=1e-3*60, margin=0.1, dilation=5):
    # Define regions
    gt_neg = gt_all - gt_pos  # Assuming gt_pos <= gt_all; clamp negatives to 0 if needed
    soft_pos = get_soft_mask(gt_pos, threshold, dilation)
    soft_neg = get_soft_mask(gt_neg, threshold, dilation)
    bg_mask = (gt_all <= threshold).float()  # Background
    
    # Normalize attentions
    norm_pos_attn = normalize_map(pos_attn)
    norm_neg_attn = normalize_map(neg_attn)
    
    # L_pos: KL alignment for positives
    L_pos = F.kl_div(norm_pos_attn.log(), soft_pos, reduction='batchmean')
    
    # L_neg: Suppress positives in negative regions + align negatives
    overlap_neg = (norm_pos_attn * soft_neg).mean()  # Penalize pos attn in neg areas
    L_neg_align = F.kl_div(norm_neg_attn.log(), soft_neg, reduction='batchmean')
    L_neg = overlap_neg + L_neg_align
    
    # L_contrast: Hinge for separation + minimize mutual in background
    diff_pos = norm_pos_attn - norm_neg_attn
    diff_neg = norm_neg_attn - norm_pos_attn
    hinge_pos = torch.relu(margin - diff_pos * soft_pos).mean()
    hinge_neg = torch.relu(margin - diff_neg * soft_neg).mean()
    mutual_bg = (norm_pos_attn * norm_neg_attn * bg_mask).mean()  # Suppress overlap in bg
    L_contrast = hinge_pos + hinge_neg + mutual_bg
    
    # Total (tune weights)
    return L_pos + L_neg + 0.5 * L_contrast

def semantic_discrimination_loss(attn_pos, attn_neg, gt_den_maps, gt_den_maps_all, alpha=1.0, beta=0.5, gamma=0.1):
    """
    语义区分损失：利用完整的标注信息进行多粒度监督
    """
    B, C, H, W = attn_pos.shape
    
    # 创建三种区域掩码
    target_mask = (gt_den_maps > 1e-3).float()           # 正语义目标区域
    other_semantic_mask = (gt_den_maps_all > 1e-3).float() - target_mask  # 其他语义的相同类别物体
    background_mask = (gt_den_maps_all == 0).float()     # 纯背景区域
    
    # 1. 正语义区域：正注意力应该高，负注意力应该低
    pos_target = (attn_pos * target_mask).flatten(1).sum(1)
    neg_target = (attn_neg * target_mask).flatten(1).sum(1)
    target_loss = F.relu(neg_target - pos_target + 0.5).mean()
    
    # 2. 其他语义区域：两者都应该中等激活（因为是同类物体）
    pos_other = (attn_pos * other_semantic_mask).flatten(1).sum(1)
    neg_other = (attn_neg * other_semantic_mask).flatten(1).sum(1)
    
    # 希望在这个区域正负注意力都有一定激活，但不要过高
    other_loss = (F.relu(0.2 - pos_other) + F.relu(0.2 - neg_other) + 
                  F.relu(pos_other - 0.8) + F.relu(neg_other - 0.8)).mean()
    
    # 3. 背景区域：两者都应该低激活
    bg_pos = (attn_pos * background_mask).flatten(1).sum(1)
    bg_neg = (attn_neg * background_mask).flatten(1).sum(1)
    bg_loss = (bg_pos + bg_neg).mean()
    
    # 4. 区分度损失：正语义区域与其他语义区域的差异
    target_mean = (attn_pos * target_mask).flatten(1).mean(1)
    other_mean = (attn_pos * other_semantic_mask).flatten(1).mean(1)
    discrimination_loss = F.relu(other_mean - target_mean + 0.3).mean()
    
    return (alpha * target_loss + beta * other_loss + 
            gamma * bg_loss + 0.5 * discrimination_loss)

def hierarchical_contrastive_loss(attn_pos, attn_neg, gt_den_maps, gt_den_maps_all, temperature=0.1):
    """
    层次化对比学习：在不同语义层次进行对比
    """
    B, C, H, W = attn_pos.shape
    
    # 区域定义
    pos_region = (gt_den_maps > 0).float()
    other_region = (gt_den_maps_all > 0).float() - pos_region
    bg_region = 1.0 - pos_region - other_region
    
    # 提取区域特征（平均激活值）
    pos_feat_pos = (attn_pos * pos_region).flatten(1).sum(1) / (pos_region.flatten(1).sum(1) + 1e-8)
    pos_feat_other = (attn_pos * other_region).flatten(1).sum(1) / (other_region.flatten(1).sum(1) + 1e-8)
    pos_feat_bg = (attn_pos * bg_region).flatten(1).sum(1) / (bg_region.flatten(1).sum(1) + 1e-8)
    
    neg_feat_pos = (attn_neg * pos_region).flatten(1).sum(1) / (pos_region.flatten(1).sum(1) + 1e-8)
    neg_feat_other = (attn_neg * other_region).flatten(1).sum(1) / (other_region.flatten(1).sum(1) + 1e-8)
    neg_feat_bg = (attn_neg * bg_region).flatten(1).sum(1) / (bg_region.flatten(1).sum(1) + 1e-8)
    
    # 层次1：正语义vs负语义在目标区域
    level1_loss = F.relu(neg_feat_pos - pos_feat_pos + 0.5).mean()
    
    # 层次2：正语义在目标区域vs其他区域
    level2_loss = F.relu(pos_feat_other - pos_feat_pos + 0.3).mean()
    
    # 层次3：负语义在目标区域vs其他区域（应该反过来）
    level3_loss = F.relu(neg_feat_pos - neg_feat_other + 0.2).mean()
    
    return level1_loss + level2_loss + level3_loss

def attention_distribution_alignment(attn_pos, attn_neg, gt_den_maps, gt_den_maps_all):
    """
    注意力分布对齐：让注意力分布与语义标注分布对齐
    """
    # 对注意力图进行softmax，得到概率分布
    attn_pos_prob = F.softmax(attn_pos.flatten(1), dim=1).view_as(attn_pos)
    attn_neg_prob = F.softmax(attn_neg.flatten(1), dim=1).view_as(attn_neg)
    
    # 对标注图进行归一化，得到目标分布
    target_dist = gt_den_maps / (gt_den_maps.flatten(1).sum(1).view(-1, 1, 1, 1) + 1e-8)
    all_dist = gt_den_maps_all / (gt_den_maps_all.flatten(1).sum(1).view(-1, 1, 1, 1) + 1e-8)
    other_dist = all_dist - target_dist
    other_dist = other_dist.clamp(min=0)
    
    # 1. 正语义注意力应该与目标分布对齐
    pos_target_align = F.kl_div(
        torch.log(attn_pos_prob + 1e-8), 
        target_dist, 
        reduction='batchmean'
    )
    
    # 2. 正语义注意力不应该与"其他语义"分布对齐
    pos_other_misalign = -F.kl_div(
        torch.log(attn_pos_prob + 1e-8),
        other_dist,
        reduction='batchmean'
    )
    
    # 3. 负语义注意力应该与"其他语义"分布有一定对齐
    neg_other_align = F.kl_div(
        torch.log(attn_neg_prob + 1e-8),
        other_dist,
        reduction='batchmean'
    )
    
    return pos_target_align + 0.5 * pos_other_misalign + 0.3 * neg_other_align

def multi_task_semantic_loss(attn_pos, attn_neg, gt_den_maps, gt_den_maps_all):
    """
    多任务语义定位损失：同时优化多个语义定位目标
    """
    B, C, H, W = attn_pos.shape
    
    # 任务1：正语义定位（主要任务）
    pos_mask = (gt_den_maps > 1e-3).float()
    task1_loss = F.binary_cross_entropy_with_logits(
        attn_pos * pos_mask,
        torch.ones_like(attn_pos) * pos_mask,
        reduction='mean'
    )
    
    # 任务2：负语义抑制
    task2_loss = F.binary_cross_entropy_with_logits(
        attn_neg * pos_mask,
        torch.zeros_like(attn_neg) * pos_mask, 
        reduction='mean'
    )
    
    # 任务3：类别一致性（所有bottle区域都应该有一定激活）
    all_bottle_mask = (gt_den_maps_all > 1e-3).float()
    task3_loss = F.mse_loss(
        torch.sigmoid(attn_pos) * all_bottle_mask,
        torch.ones_like(attn_pos) * all_bottle_mask * 0.5,  # 中等激活
        reduction='mean'
    )
    
    # 任务4：语义区分度
    pos_region_mean = (attn_pos * pos_mask).flatten(1).mean(1)
    other_region_mean = (attn_pos * (all_bottle_mask - pos_mask)).flatten(1).mean(1)
    task4_loss = F.relu(other_region_mean - pos_region_mean + 0.2).mean()
    
    return task1_loss + task2_loss + 0.3 * task3_loss + 0.5 * task4_loss

def semantic_guided_token_loss(fused_cross_attn, prompt_mask, gt_den_maps, alpha=1.0, beta=0.5):
    """
    语义引导的token注意力分配损失
    fused_cross_attn: [B, C, H, W] C=提示词token+可训练token总长度
    prompt_mask: [B, C] 提示词位置为1，可训练token位置为0
    gt_den_maps: [B, 1, H, W] 高斯热力图
    """
    B, C, H, W = fused_cross_attn.shape
    
    # 1. 分离提示词token和可训练token的注意力
    prompt_attn = fused_cross_attn * prompt_mask  # 提示词token注意力
    prompt_mask[:, 0, :, :] = 1
    trainable_attn = fused_cross_attn * (1 - prompt_mask)  # 可训练token注意力
    
    # 2. 目标区域掩码
    target_mask = (gt_den_maps > 0).float()  # 正语义目标区域
    
    # 3. 语义互补损失：可训练token应该在目标区域有更高的激活
    trainable_target_act = (trainable_attn * target_mask).flatten(2).sum(2)  # [B, C_train]
    trainable_bg_act = (trainable_attn * (1 - target_mask)).flatten(2).sum(2)  # [B, C_train]
    
    # 希望可训练token在目标区域激活 > 背景区域激活
    complement_loss = F.relu(trainable_bg_act - trainable_target_act + 0.3).mean()
    
    # 4. 语义特异性损失：可训练token应该关注提示词未能充分描述的区域
    prompt_target_act = (prompt_attn * target_mask).flatten(2).sum(2)  # [B, C_prompt]
    trainable_target_act_norm = trainable_target_act / (trainable_target_act.sum(1, keepdim=True) + 1e-8)
    
    # 如果提示词token在某些目标区域激活不足，可训练token应该补偿
    prompt_weak_regions = (prompt_target_act < prompt_target_act.mean(1, keepdim=True)).float()
    specificity_loss = (1 - (trainable_target_act_norm * prompt_weak_regions).sum(1)).mean()
    
    # 5. token协作损失：提示词和可训练token的注意力应该互补
    prompt_norm = prompt_attn.flatten(2).sum(2) / (prompt_attn.flatten(2).sum(2, keepdim=True) + 1e-8)
    trainable_norm = trainable_attn.flatten(2).sum(2) / (trainable_attn.flatten(2).sum(2, keepdim=True) + 1e-8)
    
    # 希望两者的注意力分布有一定差异（互补）
    collaboration_loss = (prompt_norm * trainable_norm).sum(1).mean()
    
    return alpha * complement_loss + beta * specificity_loss + 0.1 * collaboration_loss

def spatial_semantic_decoupling_loss(fused_cross_attn, prompt_mask, gt_den_maps, spatial_weight=2.0):
    """
    空间语义解耦损失：让可训练token专门学习空间语义信息
    """
    B, C, H, W = fused_cross_attn.shape
    
    # 分离注意力
    prompt_attn = fused_cross_attn * prompt_mask
    prompt_mask[:, 0, :, :] = 1
    trainable_attn = fused_cross_attn * (1 - prompt_mask)
    
    target_mask = (gt_den_maps > 0).float()
    
    # 1. 类别语义损失：提示词token应该负责物体识别
    # "bottle" token应该在所有bottle位置都有激活
    class_semantic_loss = F.binary_cross_entropy_with_logits(
        prompt_attn.flatten(2).max(1)[0].view(B, H, W),  # 取提示词token的最大激活
        (target_mask > 0).float().squeeze(1),
        reduction='mean'
    )
    
    # 2. 空间语义损失：可训练token应该负责空间定位
    # 计算左右半区的空间权重
    left_mask = torch.zeros(H, W, device=fused_cross_attn.device)
    left_mask[:, :W//2] = 1.0  # 左半区
    left_mask = left_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # 可训练token在左半区的激活应该更高
    trainable_left_act = (trainable_attn * left_mask * target_mask).flatten(2).sum(2)
    trainable_right_act = (trainable_attn * (1 - left_mask) * target_mask).flatten(2).sum(2)
    
    spatial_semantic_loss = F.relu(trainable_right_act - trainable_left_act + 0.5).mean()
    
    # 3. 语义解耦损失：提示词和可训练token的注意力应该解耦
    prompt_corr = torch.corrcoef(prompt_attn.flatten(1))[0,1] if B > 1 else torch.tensor(0.0)
    trainable_corr = torch.corrcoef(trainable_attn.flatten(1))[0,1] if B > 1 else torch.tensor(0.0)
    decoupling_loss = torch.abs(prompt_corr - trainable_corr)
    
    return class_semantic_loss + spatial_weight * spatial_semantic_loss + 0.1 * decoupling_loss

def dynamic_semantic_allocation_loss(fused_cross_attn, prompt_mask, gt_den_maps, temperature=0.1):
    """
    动态语义分配损失：根据注意力模式动态分配语义角色
    """
    B, C, H, W = fused_cross_attn.shape
    
    prompt_attn = fused_cross_attn * prompt_mask
    prompt_mask[:, 0, :, :] = 1
    trainable_attn = fused_cross_attn * (1 - prompt_mask)
    
    target_mask = (gt_den_maps > 0).float()
    
    # 1. 计算每个token的语义特异性得分
    prompt_specificity = (prompt_attn * target_mask).flatten(2).sum(2) / (prompt_attn.flatten(2).sum(2) + 1e-8)
    trainable_specificity = (trainable_attn * target_mask).flatten(2).sum(2) / (trainable_attn.flatten(2).sum(2) + 1e-8)
    
    # 2. 动态角色分配：可训练token应该比提示词token更专注于目标区域
    role_assignment_loss = F.relu(prompt_specificity - trainable_specificity + 0.2).mean()
    
    # 3. 语义增强损失：可训练token应该增强提示词的语义
    # 计算注意力图的互信息（简化版本）
    prompt_entropy = -torch.softmax(prompt_attn.flatten(2), dim=2) * torch.log_softmax(prompt_attn.flatten(2), dim=2)
    prompt_entropy = prompt_entropy.sum(2).mean()
    
    trainable_entropy = -torch.softmax(trainable_attn.flatten(2), dim=2) * torch.log_softmax(trainable_attn.flatten(2), dim=2)
    trainable_entropy = trainable_entropy.sum(2).mean()
    
    # 希望可训练token比提示词token更集中（熵更小）
    enhancement_loss = F.relu(trainable_entropy - prompt_entropy + 0.1).mean()
    
    # 4. 空间一致性损失：可训练token的注意力应该与空间语义一致
    left_half = torch.zeros(H, W, device=fused_cross_attn.device)
    left_half[:, :W//2] = 1.0
    left_half = left_half.unsqueeze(0).unsqueeze(0)
    
    trainable_spatial_consistency = (trainable_attn * left_half * target_mask).flatten(2).sum(2)
    trainable_spatial_inconsistency = (trainable_attn * (1 - left_half) * target_mask).flatten(2).sum(2)
    
    spatial_loss = F.relu(trainable_spatial_inconsistency - trainable_spatial_consistency + 0.3).mean()
    
    return role_assignment_loss + enhancement_loss + 2.0 * spatial_loss
