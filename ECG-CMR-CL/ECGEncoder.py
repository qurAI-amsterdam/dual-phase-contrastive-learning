# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer


class ECGEncoder(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        # ViT expects RGB input, while the ECG data only has 1 channel
        
        # kwargs['in_chans'] = 1
        kwargs['patch_size'] = (1, 100)
        print("kwargs:", kwargs)

        super(ECGEncoder, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool == "attention_pool":
            self.attention_pool = nn.MultiheadAttention(embed_dim=kwargs['embed_dim'], num_heads=kwargs['num_heads'], batch_first=True)
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            # del self.norm  # remove the original norm

    def forward_features(self, x, localized=False):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if localized:
            outcome = x[:, 1:]
        elif self.global_pool == "attention_pool":
            q = x[:, 1:, :].mean(dim=1, keepdim=True)
            k = x[:, 1:, :]
            v = x[:, 1:, :]
            x, x_weights = self.attention_pool(q, k, v) # attention pool without cls token
            outcome = self.fc_norm(x.squeeze(dim=1))
        elif self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


class ECGPredictor(nn.Module):
    def __init__(self, base_encoder, output_dim, embed_dim=1000, contrastive_learning=False):
        super().__init__()
        self.encoder = base_encoder
        self.encoder.head = nn.Identity()
        embed_dim = base_encoder.embed_dim
        self.contrastive_learning = contrastive_learning

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, output_dim)
        )

    def forward(self, x):
        if self.contrastive_learning:
            features = self.encoder(x)
        else:
            with torch.no_grad():
                features = self.encoder(x)

        features = self.classifier(features)
        return features


class ECGwClinicalPredictor(nn.Module):
    def __init__(self, base_encoder, output_dim, embed_dim=1000):
        super().__init__()
        self.encoder = base_encoder
        self.encoder.head = nn.Identity()
        embed_dim = base_encoder.embed_dim
        print(f"Embedded dimension of the base encoder = {embed_dim}")

        # self.projection = nn.Linear(embed_dim, 192)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + 12, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, output_dim)  # No activation if regression; add softmax/sigmoid as needed
        )


    def forward(self, x_ecg, x_clinical):
        with torch.no_grad():
            ecg_features = self.encoder(x_ecg)

        # ecg_features = self.projection(ecg_features)

        features = torch.cat((ecg_features, x_clinical), dim=1)

        features = self.classifier(features)
        return features


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        print("model_without_ddp", model_without_ddp)
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        print("checkpoint.keys()", checkpoint.keys())
        print("checkpoint['model'].keys()", checkpoint['model'].keys())
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def vit_pluto_patchX(**kwargs):
    model = ECGEncoder(
        embed_dim=256, depth=3, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_tiny_patchX(**kwargs):
    model = ECGEncoder(
        embed_dim=384, depth=3, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_patchX(**kwargs):
    model = ECGEncoder(
        embed_dim=512, depth=4, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_medium_patchX(**kwargs):
    model = ECGEncoder(
        embed_dim=640, depth=6, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_big_patchX(**kwargs):
    model = ECGEncoder(
        embed_dim=768, depth=8, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base_patch200(**kwargs):
    model = ECGEncoder(
        patch_size=(65, 200), embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch100(**kwargs):
    model = ECGEncoder(
        patch_size=(65, 100), embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch50(**kwargs):
    model = ECGEncoder(
        patch_size=(65, 50), embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch10(**kwargs):
    model = ECGEncoder(
        patch_size=(65, 10), embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patchX(**kwargs):
    model = ECGEncoder(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch224(**kwargs):
    model = ECGEncoder(
        patch_size=(65, 224), embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_large_patchX(**kwargs):
    model = ECGEncoder(
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch112(**kwargs):
    model = ECGEncoder(
        patch_size=(65, 112), embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_huge_patchX(**kwargs):
    model = ECGEncoder(
        embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_mediumDeep_patchX_dec256d2b(**kwargs): # nb_params: 24.68M encoder, 1.77M decoder
    model = ECGEncoder(
        embed_dim=576, depth=12, num_heads=8, # dim=64 per head
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


ECGEncoder_dict = {
    'mae_vit_mediumDeep_patchX': mae_vit_mediumDeep_patchX_dec256d2b,
    'vit_pluto_patchX': vit_pluto_patchX,
    'vit_tiny_patchX': vit_tiny_patchX,
    'vit_small_patchX': vit_small_patchX,
    'vit_medium_patchX': vit_medium_patchX,
    'vit_big_patchX': vit_big_patchX,
    'vit_base_patch200': vit_base_patch200
}