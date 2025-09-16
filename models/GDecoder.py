from functools import partial
from pathlib import Path
import torch
import torch.nn as nn
from einops import rearrange,repeat
from timm.models.vision_transformer import PatchEmbed
from models.Block.Blocks import Block
import torch.nn.functional as F
from util.pos_embed import get_2d_sincos_pos_embed
import numpy as np
from transformers import BertModel, BertTokenizer, BertConfig

class GDecoder(nn.Module):
    """ GDecoder with VisionTransformer backbone
    """
    def __init__(self, img_size=384, patch_size=16, in_chans=3, 
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, drop_path_rate = 0):
        super().__init__()
        ## Setting the model
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        ## Global Setting
        self.patch_size = patch_size
        self.img_size = img_size
        self.token_wh = img_size // patch_size
        self.norm_pix_loss = norm_pix_loss
        ## Global Setting

        ## Encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.norm = norm_layer(embed_dim)

        self.blocks = nn.ModuleList([
        Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
        for i in range(depth)])

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        ### decoder blocks
        self.decoder_depth = decoder_depth
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        ### decoder blocks
        ## Decoder specifics
        ## Regressor
        self.decode_head0 = nn.Sequential(
            nn.Conv2d(512 + self.decoder_depth, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1, stride=1)
        )  
        ## Regressor
        # self.bert = BertModel(BertConfig())
        self.bert = BertModel.from_pretrained('/data/ps/research/checkpoints/bert')
        self.tokenizer = BertTokenizer.from_pretrained('/data/ps/research/checkpoints/bert')
        self.max_text_len = 48
        self.text_pos_embed = nn.Parameter(torch.zeros(1, self.max_text_len, decoder_embed_dim), requires_grad=False)

        self.z_embed = nn.Linear(embed_dim, embed_dim, bias=True)

        self.x_embed = nn.Linear(embed_dim//2, decoder_embed_dim, bias=True)

        self.scale_embed = nn.Parameter(torch.full((1, 384, 384), fill_value=100.0), requires_grad=True)  # fixed sin-cos embedding
        
        self.encoder_norm = norm_layer(embed_dim)
        self.patch_density_embed = PatchEmbed(img_size, patch_size, 1, embed_dim)
        self.density_pos_embed = nn.Parameter(torch.zeros(1, self.patch_density_embed.num_patches, embed_dim), requires_grad=False)
        ### decoder blocks
        self.encoder_blocks = nn.ModuleList([
            Block(embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        ### decoder blocks
        
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        pos_embde_exemplar = get_2d_sincos_pos_embed(self.pos_embed_exemplar.shape[-1], int(self.patch_embed_exemplar.num_patches**.5), cls_token=False)
        self.pos_embed_exemplar.copy_(torch.from_numpy(pos_embde_exemplar).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        decoder_pos_embed_exemplar = get_2d_sincos_pos_embed(self.decoder_pos_embed_exemplar.shape[-1], int(self.patch_embed_exemplar.num_patches**.5), cls_token=False)
        self.decoder_pos_embed_exemplar.data.copy_(torch.from_numpy(decoder_pos_embed_exemplar).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w1 = self.patch_embed_exemplar.proj.weight.data
        torch.nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_w_prime():
        w = self.scale_embed.data
        w_squeezed = w.squeeze(0)
        wwT = torch.mm(w_squeezed, w_squeezed.T)
        try:
            wwT_inv = torch.inverse(wwT)
        except RuntimeError as e:
            print(e)
            wwT_inv = torch.inverse(wwT + 1e-6 * torch.eye(100, device=wwT.device))
        wT = w_squeezed.T
        w_prime = torch.mm(wT, wwT_inv)
        w_prime = w_prime.unsqueeze(0)
        return w_prime

    def sample_from_distribution(self, distribution, fact=None, sample_mean=False):

        if sample_mean:
            return distribution.loc

        # Reparameterization trick
        if fact is None:
            return distribution.rsample()

        # Resclale the eps
        eps = distribution.rsample() - distribution.loc
        latent_vector = distribution.loc + fact * eps
        return latent_vector

    def forward_text_encoder(self, text, device):
        tokenized = self.tokenizer(text, padding='max_length', truncation=True, return_tensors='pt', max_length=self.max_text_len)

        # 将 tokenized 张量移动到模型所在的设备
        tokenized = {key: value.to(device) for key, value in tokenized.items()}

        return self.bert(**tokenized).last_hidden_state
    
    def forward_img_encoder(self, x):
        
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        for i, blk in enumerate(self.blocks):
            x, attn = blk(x)
        x = self.norm(x)  
        mu_logvar = self.z_embed(x)  
        mu, logvar = torch.chunk(mu_logvar, chunks=2, dim=-1)

        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        return dist

    def forward_density_encoder(self,x):
        x = self.patch_density_embed(x.unsqueeze(1))
        # add pos embed
        x = x + self.density_pos_embed
        
        for i, blk in enumerate(self.encoder_blocks):
            x, attn = blk(x)
        x = self.encoder_norm(x)
        mu_logvar = self.z_embed(x)  
        mu, logvar = torch.chunk(x, chunks=2, dim=-1)

        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        return dist
    
    def forward_decoder(self,x,y):
        x = self.x_embed(x)
        # add pos embed
        x = x + self.decoder_pos_embed
        b,l_x,d = x.shape
        
        y = self.decoder_embed(y) + self.text_pos_embed
        x = torch.cat((x,y),axis=1)
        attns = []
        xs = []
        ys = []
        for i, blk in enumerate(self.decoder_blocks):
            x, attn = blk(x)
            if i == self.decoder_depth-1:
                x = self.decoder_norm(x)
            attns.append(attn)
            xs.append(x[:,:l_x,:])
            ys.append(x[:,l_x:,:])
        return xs,ys,attns

    def Regressor(self, feature):
        feature = F.interpolate(
                                self.decode_head0(feature), size=feature.shape[-1]*2, mode='bilinear', align_corners=False)
        feature = F.interpolate(
                                self.decode_head1(feature), size=feature.shape[-1]*2, mode='bilinear', align_corners=False)
        feature = F.interpolate(
                                self.decode_head2(feature), size=feature.shape[-1]*2, mode='bilinear', align_corners=False)
        feature = F.interpolate(
                                self.decode_head3(feature), size=feature.shape[-1]*2, mode='bilinear', align_corners=False)
        feature = feature.squeeze(-3)
        return feature
    
    def A2F(self, xs, attns):
        density_feature = rearrange(xs[-1],'b (w h) d->b d w h',w = self.token_wh)

        attn2feat = []
        for attn in attns:
            attns_x2y = nn.Sigmoid()(attn[:, :, self.num_patches:, :self.num_patches])
            attns_x2y = torch.mean(attns_x2y,dim=1)
            attns_x2y = attns_x2y.mean(dim=1).unsqueeze(1)
            attns_x2y = rearrange(attns_x2y,'b c (w h)->b c w h',w = self.token_wh, h = self.token_wh)
            attn2feat.append(attns_x2y)
        attn2feat = torch.stack(attn2feat, dim=0)
        attn2feat = rearrange(attn2feat,'n b c w h->b (n c) w h')
        density_feature = torch.cat((density_feature.contiguous(), attn2feat.contiguous()), axis=1)

        return density_feature

    def forward_density2density(self, x, t): 
        dist = self.forward_density_encoder(x)
        z = self.sample_from_distribution(dist)
        
        xs,ys,attns = self.forward_decoder(z,t)

        density_feature = self.A2F(xs, attns)

        density_map = self.Regressor(density_feature)
        w_prime = get_w_prime()

        density_map = torch.einsum("bhw,wh->bhh", density_map, w_prime)
        return density_map, dist, z


    def forward_img2density(self, x, t): 
        dist = self.forward_img_encoder(x)
        z = self.sample_from_distribution(dist)
        
        xs,ys,attns = self.forward_decoder(z,t)

        density_feature = self.A2F(xs, attns)

        density_map = self.Regressor(density_feature)
        w_prime = get_w_prime()

        density_map = torch.einsum("bhw,wh->bhh", density_map, w_prime)
        
        return density_map, dist, z
    
    def forward(self, samples,name = None): 
        imgs = samples[0]
        text = samples[1]
        text_token = self.forward_text_encoder(text, imgs.device)

        density_map_i, dist_i, z_i = self.forward_img2density(imgs, text_token)
        

        if self.training:
            gtmap = torch.einsum("bhw,wh->bhh", samples[2], self.scale_embed)
            gtmap = samples[2] * self.scale_embed.repeat(imgs.shape[0], 1, 1)
            density_map_d, dist_d, z_d = self.forward_density2density(gtmap, text_token)
            mu_ref = torch.zeros_like(dist_d.loc)
            scale_ref = torch.ones_like(dist_d.scale)
            dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
            return density_map_i, dist_i, dist_d, dist_ref, density_map_d, gtmap, z_i, z_d
        else:
            return density_map_i

def model_gdecoder(**kwargs):
    model_base = GDecoder(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model_base

    
