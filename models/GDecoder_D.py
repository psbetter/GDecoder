from functools import partial
from pathlib import Path
import torch
import torch.nn as nn
from einops import rearrange,repeat
from timm.models.vision_transformer import PatchEmbed
from models.Blocks import Block, get_2d_sincos_pos_embed
import torch.nn.functional as F
import numpy as np
from transformers import BertModel, BertTokenizer, BertConfig
from models.decoder import Regressor

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
        self.Regressor = Regressor(512 + self.decoder_depth) 
        ## Regressor
        # self.bert = BertModel(BertConfig())
        self.bert = BertModel.from_pretrained('/home/ubuntu/jyt/checkpoints/bert')
        self.tokenizer = BertTokenizer.from_pretrained('/home/ubuntu/jyt/checkpoints/bert')

        for param in self.bert.parameters():
            param.requires_grad = False

        self.max_text_len = 48
        self.text_pos_embed = nn.Parameter(torch.zeros(1, self.max_text_len, decoder_embed_dim), requires_grad=False)

        self.z_embed = nn.Linear(embed_dim, embed_dim, bias=True)

        self.x_embed = nn.Linear(embed_dim//2, decoder_embed_dim, bias=True)

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
        
        # pos_embde_exemplar = get_2d_sincos_pos_embed(self.pos_embed_exemplar.shape[-1], int(self.patch_embed_exemplar.num_patches**.5), cls_token=False)
        # self.pos_embed_exemplar.copy_(torch.from_numpy(pos_embde_exemplar).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # decoder_pos_embed_exemplar = get_2d_sincos_pos_embed(self.decoder_pos_embed_exemplar.shape[-1], int(self.patch_embed_exemplar.num_patches**.5), cls_token=False)
        # self.decoder_pos_embed_exemplar.data.copy_(torch.from_numpy(decoder_pos_embed_exemplar).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # w1 = self.patch_embed_exemplar.proj.weight.data
        # torch.nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
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

    def forward_text_encoder(self, text, device):
        # print(text)
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
        feat = self.z_embed(x)  
        return feat

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

    def forward_img2density(self, x, t): 
        feat = self.forward_img_encoder(x)
        
        xs,ys,attns = self.forward_decoder(feat,t)

        density_feature = self.A2F(xs, attns)

        density_map = self.Regressor(density_feature)
        
        return density_map
    
    def forward(self, samples,name = None): 
        imgs = samples[0]
        text = samples[1]
        text_token = self.forward_text_encoder(text, imgs.device)

        density_map_i = self.forward_img2density(imgs, text_token)
        return density_map_i

def model_gdecoder(**kwargs):
    model_base = GDecoder(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model_base

    
