import torch.nn as nn
import torch
import math
import torch.nn.functional as F


class Regressor(nn.Module):
    def __init__(self, out_dim):
        super(Regressor, self).__init__()
        self.conv = nn.Sequential(*[nn.Conv2d(out_dim, out_dim, 3, 1, padding=1), nn.ReLU()])
        self.img_pe_transform1 = nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=True, groups=out_dim)
        self.self_attn1 = Attention(out_dim, 8)
        self.norm1 = nn.LayerNorm(out_dim)
        self.img_pe_transform2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=True, groups=out_dim)
        self.self_attn2 = Attention(out_dim, 8)
        self.norm2 = nn.LayerNorm(out_dim)

        self.reg_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(out_dim, out_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(out_dim, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        B, _, H, W = x.shape
        x_pe = self.img_pe_transform1(x)
        x = x.flatten(2).transpose(1, 2)
        x_pe = x_pe.flatten(2).transpose(1, 2)
        attn_out = self.self_attn1(q=x + x_pe, k=x + x_pe, v=x)
        x = x + attn_out
        x = self.norm1(x)
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = self.conv(x)
        x_pe = self.img_pe_transform2(x)
        x = x.flatten(2).transpose(1, 2)
        x_pe = x_pe.flatten(2).transpose(1, 2)
        attn_out = self.self_attn2(q=x + x_pe, k=x + x_pe, v=x)
        x = x + attn_out
        x = self.norm2(x)
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = self.reg_head(x)
        return x


class Upsample2(nn.Module):
    def __init__(self, up_in_ch, up_out_ch, cat_in_ch, cat_out_ch, sem=True):
        super(Upsample2, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Sequential(*[nn.Conv2d(up_in_ch, up_out_ch, 3, 1, padding=1), nn.ReLU()])
        self.conv2 = nn.Sequential(*[nn.Conv2d(cat_in_ch, cat_out_ch, 3, 1, padding=1), nn.ReLU()])

        self.text_conv = nn.Linear(768, cat_out_ch)
        self.norm1 = nn.LayerNorm(cat_out_ch)
        self.txt_attn = ImgTxtFusion2(cat_out_ch, 8)
        self.text_conv2 = nn.Linear(768, cat_out_ch)
        self.norm2 = nn.LayerNorm(cat_out_ch)
        self.img_attn = ImgTxtFusion2(cat_out_ch, 8)
        # self.conv3 = nn.Sequential(*[nn.Conv2d(cat_out_ch, cat_out_ch, 3, 1, padding=1), nn.ReLU()])

    def forward(self, low, high, text_feat, latents_feat):
        low = self.up(low)
        low = self.conv1(low)
        x = torch.cat([high, low], dim=1)
        B, _, H, W = x.shape
        x = self.conv2(x).flatten(2).transpose(1, 2)

        text_feat = self.text_conv(text_feat)
        text_feat = text_feat + self.norm1(text_feat)  # B, C
        # print(text_feat.shape)
        # print(x.shape)
        text_feat  = self.txt_attn(text_feat, x)
        
        latents_feat = self.text_conv2(latents_feat)
        latents_feat = latents_feat + self.norm2(latents_feat)  # B, C
        x = self.img_attn(x, latents_feat)

        x = self.img_attn(x, text_feat)

        x = x.transpose(1, 2).view(B, -1, H, W)
        return x

class Upsample(nn.Module):
    def __init__(self, up_in_ch, up_out_ch, cat_in_ch, cat_out_ch, sem=True):
        super(Upsample, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Sequential(*[nn.Conv2d(up_in_ch, up_out_ch, 3, 1, padding=1), nn.ReLU()])
        self.conv2 = nn.Sequential(*[nn.Conv2d(cat_in_ch, cat_out_ch, 3, 1, padding=1), nn.ReLU()])

    def forward(self, low, high):
        low = self.up(low)
        low = self.conv1(low)
        x = torch.cat([high, low], dim=1)
        x = self.conv2(x)
        return x


class Attention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        assert self.embedding_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

    def _separate_heads(self, x, num_heads):
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x):
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q, k, v):
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class ImgTxtFusion(nn.Module):
    def __init__(self, channel_num, num_head):
        super(ImgTxtFusion, self).__init__()
        self.img_pe_transform = nn.Conv2d(channel_num, channel_num, 3, 1, 1, bias=True, groups=channel_num)
        self.txt2img = Attention(channel_num, num_head)
        self.norm1 = nn.LayerNorm(channel_num)
        self.mlp = FeedForward(channel_num)
        self.norm2 = nn.LayerNorm(channel_num)
        self.img2txt = Attention(channel_num, num_head)
        self.norm3 = nn.LayerNorm(channel_num)

    def forward(self, img_feat, txt_feat):
        img_pe = self.img_pe_transform(img_feat)
        B, _, H, W = img_pe.shape
        img_pe = img_pe.flatten(2).transpose(1, 2)  # B, N, C
        img_feat = img_feat.flatten(2).transpose(1, 2)  # B, N, C
        txt_feat = txt_feat.unsqueeze(1) # B, 1, C

        attn_out = self.txt2img(q=txt_feat, k=img_feat+img_pe, v=img_feat)
        txt_feat_ = txt_feat + attn_out
        txt_feat_ = self.norm1(txt_feat_)

        mlp_out = self.mlp(txt_feat_)
        txt_feat_ = txt_feat_ + mlp_out
        txt_feat_ = self.norm2(txt_feat_)

        attn_out = self.img2txt(q=img_feat+img_pe, k=txt_feat_+txt_feat, v=txt_feat_)
        img_feat = img_feat + attn_out
        img_feat = self.norm3(img_feat)

        return txt_feat_.squeeze(1), img_feat.transpose(1, 2).reshape(B, -1, H, W)

class ImgTxtFusion2(nn.Module):
    def __init__(self, channel_num, num_head):
        super(ImgTxtFusion2, self).__init__()
        self.attn = Attention(channel_num, num_head)
        self.norm = nn.LayerNorm(channel_num)
        self.mlp = FeedForward(channel_num)

    def forward(self, img_feat, txt_feat):

        attn_out = self.attn(q=img_feat, k=txt_feat, v=txt_feat)
        img_feat = img_feat + attn_out
        img_feat = self.norm(img_feat)

        return img_feat


# 4 8 16 32 x
class DensityRegressor(nn.Module):
    def __init__(self, counter_dim):
        super().__init__()
        # 1/32 -> 1/16
        self.conv0 = nn.Sequential(nn.Conv2d(counter_dim * 2, counter_dim, 7, padding=3),
                                   nn.ReLU())
        # 1/16 -> 1/8
        self.conv1 = nn.Sequential(nn.Conv2d(counter_dim * 3, counter_dim, 5, padding=2),
                                   nn.ReLU())
        # 1/8 -> 1/4
        self.conv2 = nn.Sequential(nn.Conv2d(counter_dim * 3, counter_dim, 3, padding=1),
                                   nn.ReLU())
        # 1/4 -> 1/2
        self.conv3 = nn.Sequential(nn.Conv2d(counter_dim * 3, counter_dim, 3, padding=1),
                                   nn.ReLU())
        # 1/2 -> 1
        # self.up2x = nn.Sequential(
        #                 nn.Conv2d(counter_dim, counter_dim//2, 1),
        #                 nn.ReLU(),)
        self.up2x = nn.Sequential(
                                nn.Conv2d(counter_dim, counter_dim//2, 3, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(counter_dim//2, counter_dim//4, 1),
                                nn.ReLU(),)
        self.conv4 = nn.Sequential(
                                nn.Conv2d(counter_dim//4, 1, 1),
                                nn.ReLU())
        # self.conv4 = nn.Conv2d(counter_dim//2, 1, 1)
        # self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        
        self.down2x = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down4x = nn.MaxPool2d(kernel_size=4, stride=4, ceil_mode=True)
        self.down8x = nn.MaxPool2d(kernel_size=8, stride=8, ceil_mode=True)
        
        self._weight_init_()
        
    def forward(self, features, cnns, img_shape = [1000,1000], hidden_output=False):
        ## multi-layer context-aware density feature encode
        # f = features[3] * cnns[3]
        x = torch.cat([features[3], cnns[3]], dim=1)
        x1 = self.conv0(x)

        x = F.interpolate(x1, size = features[2].shape[-2:], mode='bilinear')
        # x = self.pixel_shuffle(x1) ##
        # f = features[2] * cnns[2]
        x = torch.cat([x, features[2], cnns[2]], dim=1)
        # smap = F.interpolate(smaps[0], size = features[2].shape[-2:], mode='bilinear')

        x2 = self.conv1(x)

        x = F.interpolate(x2, size = features[1].shape[-2:], mode='bilinear')
        x = torch.cat([x, features[1], cnns[1]], dim=1)
        
        #smap = F.interpolate(smaps[0], size = features[1].shape[-2:], mode='bilinear')
        # x = x
        x3 = self.conv2(x)

        x = F.interpolate(x3, size = features[0].shape[-2:], mode='bilinear')

        x = torch.cat([x, features[0], cnns[0]], dim=1)
        # x = x
        x4 = self.conv3(x)
        xx4 = x4
        ## down-scale density feature
        xx3 = self.down2x(x4)
        xx2 = self.down4x(x4)
        xx1 = self.down8x(x4)
        
        ## multi-layer context-aware density feature decode
        
        x = self.up2x(x4)

        x = F.interpolate(x, size = img_shape, mode='bilinear')
        
        x = self.conv4(x)
        
        # x = F.sigmoid(x)
        # x = x * 60
        hidden_mode = 'fpn'
        if hidden_output:
            if hidden_mode == 'down':
                return x, [xx4,xx3,xx2,xx1]
            if hidden_mode == 'fpn':
                return x, [x4,x3,x2,x1]
        else:
            return x

    def _weight_init_(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.01, std=0.01)
                # nn.init.kaiming_uniform_(
                #         m.weight, 
                #         mode='fan_in', 
                #         nonlinearity='relu'
                #         )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
