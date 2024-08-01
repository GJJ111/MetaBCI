# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/7/06
# License: MIT License
"""
ShallowFBCSP.
Modified from https://github.com/braindecode/braindecode/blob/master/braindecode/models/shallow_fbcsp.py

"""

from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor
from .base import SkorchNet
from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from .util.pos_embed import get_2d_sincos_pos_embed

from .util.objectives import cca_loss

import numpy as np

import numpy


import argparse

class MaskedAutoencoderViT_py(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, num_class=5, 
                 fea_pos_mode=3, args=None):
        super().__init__()
        
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_norm=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_norm=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        if isinstance(patch_size, tuple): 
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size[0] * patch_size[1] * in_chans, bias=True) # decoder to patch
        else:
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.classify = nn.Linear(embed_dim, num_class)
        # --------------------------------------------------------------------------

        # 1 只有fea
        # 2 只有pos
        # 3 fea + pos
        self.fea_pos_mode = fea_pos_mode
        
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        grid_h = self.patch_embed.img_size[0] // self.patch_embed.patch_size[0]
        grid_w = self.patch_embed.img_size[1] // self.patch_embed.patch_size[1]
        
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (grid_h, grid_w), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], (grid_h, grid_w), cls_token=True)
        # decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

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

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        _, chs, img_h, img_w = imgs.shape
        if isinstance(self.patch_embed.patch_size, tuple):
            p_h = self.patch_embed.patch_size[0]
            p_w = self.patch_embed.patch_size[1]
            h = img_h // p_h
            w = img_w // p_w
        else:
            p_h = p_w = self.patch_embed.patch_size
            h = w = self.patch_embed.patch_size // p_h
        #assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        
        x = imgs.reshape(shape=(imgs.shape[0], chs, h, p_h, w, p_w))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p_h * p_w * chs))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        #x = x.unsqueeze(3)
        x = self.patch_embed(x) # (60, 256) --> (12 x 5, 16, 16)
        #print(x.shape)

        # add pos embed w/o cls token
        
        if self.fea_pos_mode == 1:
            pass
        elif self.fea_pos_mode == 2:
            x = self.pos_embed[:, 1:, :].repeat((x.shape[0], 1, 1))
        elif self.fea_pos_mode == 3:
            x = x + self.pos_embed[:, 1:, :]
        else:
            assert False, "self.fea_pos_mode 取值范围应为1，2，3"
        
        # masking: length -> length * mask_ratio
        # 会乱序？
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
            
        
        # x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        if mask.sum() > 0:
            return (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        else:
            return loss.sum()
        

    def forward(self, imgs, label=None, args=None, mask_ratio=0.75):
        imgs = imgs.unsqueeze(1)
        
        if args.training_type == 'pretrianing':
            latent, mask, ids_restore = self.forward_encoder(imgs, args, mask_ratio)
            pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
            loss = self.forward_loss(imgs, pred, mask)
        elif args.training_type == 'supervised':
            latent, mask, ids_restore = self.forward_encoder(imgs, args, 0)
            pred = self.classify(latent.mean(dim=1))
            
            criterion = torch.nn.CrossEntropyLoss()
            loss = None
            if label is not None:
                loss = criterion(pred, label)
            
        return loss, pred, mask
    
@SkorchNet  # TODO: Bug Fix required:  unable to make docs with this wrapper
class MetaBciTransformer(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=(64,5), patch_size=16, in_chans=3,
                 embed_dim=16, depth=1, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, num_class=5, **kwargs):
        super().__init__()

        
        fpnt = img_size[1]
        #print(fpnt)#176
        
        self.model_f = MaskedAutoencoderViT_py(
            img_size=(64, 5), patch_size=(1, 5), in_chans=1, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16, num_class=num_class, 
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        
        self.model_c = MaskedAutoencoderViT_py(
            img_size=(64, 5), patch_size=(64, 1), in_chans=1, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16, num_class=num_class, 
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        
        #在此处加最后一层transformer编码器！！！！！！！！！！！！看一下输出的序列size是多少，有几个序列，定义patch_size
        self.model_t=MaskedAutoencoderViT_py(
            img_size=(20, embed_dim*2), patch_size=(1, embed_dim*2), in_chans=1, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16, num_class=num_class, 
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

        # MAE decoder specifics
        self.classify = nn.Linear(embed_dim, num_class)
        self.regress = nn.Linear(embed_dim, 1)
        
        self.training_type = None


    def forward(self, imgs, label=None, args=None):
        
        
        #imgs = imgs.unsqueeze(1)#64*60*176->64*1*60*176
        #print(12345)
        #print(imgs.shape)
        
        
        T=[]
        for i in range(imgs.shape[1]):
        
            t=imgs[:,i,:,:]
            #print(t.shape)
            t = t.unsqueeze(1)#64*60*176->64*1*60*176
            #print(t.shape)
            #
            latent_f, mask, ids_restore = self.model_f.forward_encoder(t, 0)#
            latent_c, mask, ids_restore = self.model_c.forward_encoder(t, 0)
            #latent_f, mask, ids_restore = self.model_f.forward_encoder(t, 0)#
            #latent_c, mask, ids_restore = self.model_c.forward_encoder(t, 0)
            #print(latent_f.shape)#64.61.512
            #print(latent_c.shape)#64.177.512
            latent = torch.cat([latent_f.mean(dim=1), latent_c.mean(dim=1)], dim=1)#64*1024
            #print(latent.shape)
            T.append(latent)
            result = torch.stack(T, axis=1)
        #print(result.shape)
        result = result.unsqueeze(1)
        #print(result.shape)
        latent_t, mask, ids_restore = self.model_t.forward_encoder(result,0) 
        #latent_t, mask, ids_restore = self.model_t.forward_encoder(result, 0)    
        pred = self.classify(latent_t.mean(dim=1))#.mean(dim=1))[:,0,:]
        
        latent_final = latent_t
        # regress = self.regress(latent_final).squeeze() # （批次，窗口）
        
        # if len(regress.shape) < 2:
        #     regress = regress.unsqueeze(0)
        
        criterion = torch.nn.CrossEntropyLoss()##换一个损失！！！！！！
        loss = None
        if label is not None:
            loss = criterion(pred, label)
        #print(latent.shape)#64*1024查看张量大小
        return  pred
            #return loss, pred, mask
