# coding: utf-8
""" BigGAN PyTorch model.
    From "Large Scale GAN Training for High Fidelity Natural Image Synthesis"
    By Andrew Brocky, Jeff Donahuey and Karen Simonyan.
    https://openreview.net/forum?id=B1xsqj09Fm

    PyTorch version implemented from the computational graph of the TF Hub module for BigGAN.
    Some part of the code are adapted from https://github.com/brain-research/self-attention-gan

    This version only comprises the generator (since the discriminator's weights are not released).
    This version only comprises the "deep" version of BigGAN (see publication).
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

sys.path.append("..")

from model.utils.biggan_config import BigGANConfig
from model.utils.biggan_file_utils import cached_path

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    "biggan-deep-128": "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-128-pytorch_model.bin",
    "biggan-deep-256": "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-256-pytorch_model.bin",
    "biggan-deep-512": "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-512-pytorch_model.bin",
}

PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "biggan-deep-128": "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-128-config.json",
    "biggan-deep-256": "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-256-config.json",
    "biggan-deep-512": "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-512-config.json",
}

WEIGHTS_NAME = "pytorch_model.bin"
CONFIG_NAME = "config.json"


def snconv2d(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Conv2d(**kwargs), eps=eps)


def snlinear(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Linear(**kwargs), eps=eps)


def sn_embedding(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Embedding(**kwargs), eps=eps)


class SelfAttn(nn.Module):
    """Self attention Layer"""

    def __init__(self, in_channels, eps=1e-12):
        super(SelfAttn, self).__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(
            in_channels=in_channels,
            out_channels=in_channels // 8,
            kernel_size=1,
            bias=False,
            eps=eps,
        )
        self.snconv1x1_phi = snconv2d(
            in_channels=in_channels,
            out_channels=in_channels // 8,
            kernel_size=1,
            bias=False,
            eps=eps,
        )
        self.snconv1x1_g = snconv2d(
            in_channels=in_channels,
            out_channels=in_channels // 2,
            kernel_size=1,
            bias=False,
            eps=eps,
        )
        self.snconv1x1_o_conv = snconv2d(
            in_channels=in_channels // 2,
            out_channels=in_channels,
            kernel_size=1,
            bias=False,
            eps=eps,
        )
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch // 8, h * w)
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch // 8, h * w // 4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch // 2, h * w // 4)
        # Attn_g - o_conv
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch // 2, h, w)
        attn_g = self.snconv1x1_o_conv(attn_g)
        # Out
        out = x + self.gamma * attn_g
        return out


class BigGANBatchNorm(nn.Module):
    """This is a batch norm module that can handle conditional input and can be provided with pre-computed
    activation means and variances for various truncation parameters.

    We cannot just rely on torch.batch_norm since it cannot handle
    batched weights (pytorch 1.0.1). We computate batch_norm our-self without updating running means and variances.
    If you want to train this model you should add running means and variance computation logic.
    """

    def __init__(
        self,
        num_features,
        condition_vector_dim=None,
        n_stats=51,
        eps=1e-4,
        conditional=True,
    ):
        super(BigGANBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.conditional = conditional

        # We use pre-computed statistics for n_stats values of truncation between 0 and 1
        self.register_buffer("running_means", torch.zeros(n_stats, num_features))
        self.register_buffer("running_vars", torch.ones(n_stats, num_features))
        self.step_size = 1.0 / (n_stats - 1)

        if conditional:
            assert condition_vector_dim is not None
            self.scale = snlinear(
                in_features=condition_vector_dim,
                out_features=num_features,
                bias=False,
                eps=eps,
            )
            self.offset = snlinear(
                in_features=condition_vector_dim,
                out_features=num_features,
                bias=False,
                eps=eps,
            )
        else:
            self.weight = torch.nn.Parameter(torch.Tensor(num_features))
            self.bias = torch.nn.Parameter(torch.Tensor(num_features))

    def forward(self, x, truncation, condition_vector=None):
        # Retreive pre-computed statistics associated to this truncation
        coef, start_idx = math.modf(truncation / self.step_size)
        start_idx = int(start_idx)
        if coef != 0.0:  # Interpolate
            running_mean = self.running_means[start_idx] * coef + self.running_means[
                start_idx + 1
            ] * (1 - coef)
            running_var = self.running_vars[start_idx] * coef + self.running_vars[
                start_idx + 1
            ] * (1 - coef)
        else:
            running_mean = self.running_means[start_idx]
            running_var = self.running_vars[start_idx]

        if self.conditional:
            running_mean = running_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            running_var = running_var.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            weight = 1 + self.scale(condition_vector).unsqueeze(-1).unsqueeze(-1)
            bias = self.offset(condition_vector).unsqueeze(-1).unsqueeze(-1)

            out = (x - running_mean) / torch.sqrt(
                running_var + self.eps
            ) * weight + bias
        else:
            out = F.batch_norm(
                x,
                running_mean,
                running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=0.0,
                eps=self.eps,
            )

        return out


class GenBlock(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        condition_vector_dim,
        reduction_factor=4,
        up_sample=False,
        n_stats=51,
        eps=1e-12,
    ):
        super(GenBlock, self).__init__()
        self.up_sample = up_sample
        self.drop_channels = in_size != out_size
        middle_size = in_size // reduction_factor

        self.bn_0 = BigGANBatchNorm(
            in_size, condition_vector_dim, n_stats=n_stats, eps=eps, conditional=True
        )
        self.conv_0 = snconv2d(
            in_channels=in_size, out_channels=middle_size, kernel_size=1, eps=eps
        )

        self.bn_1 = BigGANBatchNorm(
            middle_size,
            condition_vector_dim,
            n_stats=n_stats,
            eps=eps,
            conditional=True,
        )
        self.conv_1 = snconv2d(
            in_channels=middle_size,
            out_channels=middle_size,
            kernel_size=3,
            padding=1,
            eps=eps,
        )

        self.bn_2 = BigGANBatchNorm(
            middle_size,
            condition_vector_dim,
            n_stats=n_stats,
            eps=eps,
            conditional=True,
        )
        self.conv_2 = snconv2d(
            in_channels=middle_size,
            out_channels=middle_size,
            kernel_size=3,
            padding=1,
            eps=eps,
        )

        self.bn_3 = BigGANBatchNorm(
            middle_size,
            condition_vector_dim,
            n_stats=n_stats,
            eps=eps,
            conditional=True,
        )
        self.conv_3 = snconv2d(
            in_channels=middle_size, out_channels=out_size, kernel_size=1, eps=eps
        )

        self.relu = nn.ReLU()

    def forward(self, x, cond_vector, truncation):
        x0 = x

        x = self.bn_0(x, truncation, cond_vector)
        x = self.relu(x)
        x = self.conv_0(x)

        x = self.bn_1(x, truncation, cond_vector)
        x = self.relu(x)
        if self.up_sample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv_1(x)

        x = self.bn_2(x, truncation, cond_vector)
        x = self.relu(x)
        x = self.conv_2(x)

        x = self.bn_3(x, truncation, cond_vector)
        x = self.relu(x)
        x = self.conv_3(x)

        if self.drop_channels:
            new_channels = x0.shape[1] // 2
            x0 = x0[:, :new_channels, ...]
        if self.up_sample:
            x0 = F.interpolate(x0, scale_factor=2, mode="nearest")

        out = x + x0
        return out


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        intermediate = []
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            intermediate.append(x)
        return x, intermediate[-2]


class LatentClassfier(nn.Module):
    def __init__(
        self,
        size,
        in_channels,
        conv_down,
        hidden_dim,
        output_dim,
        num_layers,
    ):
        super().__init__()
        self.conv_down = nn.Conv2d(in_channels, conv_down, 1, bias=False)
        # input_dim = conv_down * size**2
        input_dim = size * conv_down
        # (((W - K + 2P)/S) + 1)
        # TODO: compute hidden_dim
        self.mlp = MLP(input_dim, hidden_dim, output_dim, num_layers)

    def forward(self, z):
        in_mlp = self.conv_down(z)
        in_mlp = in_mlp.flatten(1)
        out_mlp = self.mlp(in_mlp)
        return out_mlp


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        ch = config.channel_width
        condition_vector_dim = config.z_dim * 2

        self.gen_z = snlinear(
            in_features=condition_vector_dim,
            out_features=4 * 4 * 16 * ch,
            eps=config.eps,
        )

        layers = []
        for i, layer in enumerate(config.layers):
            if i == config.attention_layer_position:
                layers.append(SelfAttn(ch * layer[1], eps=config.eps))
            layers.append(
                GenBlock(
                    ch * layer[1],
                    ch * layer[2],
                    condition_vector_dim,
                    up_sample=layer[0],
                    n_stats=config.n_stats,
                    eps=config.eps,
                )
            )

        self.layers = nn.ModuleList(layers)

        self.bn = BigGANBatchNorm(
            ch, n_stats=config.n_stats, eps=config.eps, conditional=False
        )
        self.relu = nn.ReLU()
        self.conv_to_rgb = snconv2d(
            in_channels=ch, out_channels=ch, kernel_size=3, padding=1, eps=config.eps
        )
        self.tanh = nn.Tanh()

        # Freezing base GAN
        if config.freeze_gan:
            print("Freezing GAN params")
            for p in self.parameters():
                p.requires_grad = False

        # Classifiers
        self.attention_layer_position = config.attention_layer_position

        clf = config.clf
        self.clf_on = clf["on"]
        latent_clf = []
        if self.clf_on:
            self.alpha = clf["alpha"]
            # TODO: find how to dynamically fecth size
            res = [4, 8, 8, 16, 16, 32, 32, 64, 64, 64, 128, 128, 256]
            hidden_dims = [
                2048,
                2048,
                2048,
                1024,
                1024,
                1024,
                1024,
                512,
                512,
                512,
                256,
                256,
                128,
            ]
            for i, (layer, r) in enumerate(zip(layers, res)):
                if i != config.attention_layer_position:
                    size = r**2
                    latent_clf.append(
                        LatentClassfier(
                            size=size,
                            in_channels=layer.conv_3.out_channels,
                            conv_down=clf["conv_down"],
                            hidden_dim=hidden_dims[i],
                            output_dim=clf["output_dim"],
                            num_layers=clf["num_layers"],
                        )
                    )
                else:
                    latent_clf.append(nn.Identity())
            self.latent_clf = nn.ModuleList(latent_clf)

    def forward(self, cond_vector, truncation):
        out_clf = []
        z = self.gen_z(cond_vector)  # [n, 32768]
        # We use this conversion step to be able to use TF weights:
        # TF convention on shape is [batch, height, width, channels]
        # PT convention on shape is [batch, channels, height, width]
        z = z.view(-1, 4, 4, 16 * self.config.channel_width)  # [n, 4, 4, 2048]

        z = z.permute(0, 3, 1, 2).contiguous()  # [5, 2048, 4, 4]

        if self.clf_on:
            for i, (layer, clf) in enumerate(
                zip(self.layers, self.latent_clf)
            ):  # 其中有一层是 self-attention
                if isinstance(layer, GenBlock):
                    z = layer(z, cond_vector, truncation)  # [5, 2048, 4, 4]
                else:
                    z = layer(z)  # [5, 2048, 8, 8]
                if i != self.attention_layer_position:
                    x, feat_fusion = clf(z)
                    B, C = feat_fusion.shape
                    z = z + self.alpha * feat_fusion.clone().detach().reshape(B, C, 1, 1)
                    out_clf.append(x)
        else:
            for layer in self.layers:
                if isinstance(layer, GenBlock):
                    z = layer(z, cond_vector, truncation)  # [5, 2048, 4, 4]
                else:
                    z = layer(z)  # [5, 2048, 8, 8]

        z = self.bn(z, truncation)  # [5, 128, 256, 256]

        z = self.relu(z)

        z = self.conv_to_rgb(z)  # [5, 128, 256, 256]

        z = z[:, :3, ...]  # [5, 3, 256, 256]

        z_final = self.tanh(z)
        # TODO: clf return here
        return z_final if not self.clf_on else (z_final, out_clf)  # [5, 3, 256, 256]


class BigGAN(nn.Module):
    """BigGAN Generator."""

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, cache_dir=None, *inputs, **kwargs
    ):
        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            model_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
            config_file = PRETRAINED_CONFIG_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            model_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)

        try:
            resolved_model_file = cached_path(model_file, cache_dir=cache_dir)
            resolved_config_file = cached_path(config_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Wrong model name, should be a valid path to a folder containing "
                "a {} file and a {} file or a model name in {}".format(
                    WEIGHTS_NAME, CONFIG_NAME, PRETRAINED_MODEL_ARCHIVE_MAP.keys()
                )
            )
            raise

        logger.info(
            "loading model {} from cache at {}".format(
                pretrained_model_name_or_path, resolved_model_file
            )
        )

        # Load config
        config = BigGANConfig.from_json_file(resolved_config_file)

        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        state_dict = torch.load(
            resolved_model_file,
            map_location="cpu" if not torch.cuda.is_available() else None,
        )
        model.load_state_dict(state_dict, strict=False)
        return model

    def __init__(self, config):
        super(BigGAN, self).__init__()
        self.config = config
        self.embeddings = nn.Linear(config.num_classes, config.z_dim, bias=False)
        if config.freeze_gan:
            for p in self.embeddings.parameters():
                p.requires_grad = False
        self.generator = Generator(config)

    def forward(self, z, class_label, truncation):
        assert 0 < truncation <= 1

        embed = self.embeddings(class_label)
        # print(embed.shape) # 1000->128
        cond_vector = torch.cat((z, embed), dim=1)  # 128->256

        z = self.generator(cond_vector, truncation)
        return z, cond_vector


if __name__ == "__main__":
    import PIL
    from pytorch_pretrained_biggan.utils import (
        truncated_noise_sample,
        save_as_images,
        one_hot_from_names,
    )
    from pytorch_pretrained_biggan.convert_tf_to_pytorch import (
        load_tf_weights_in_biggan,
    )

    load_tf = False
    cache_path = "../checkpoint/biggan/256/G-256.pt"
    resolved_config_file = "../checkpoint/biggan/256/biggan-deep-256-config.json"
    config = BigGANConfig.from_json_file(resolved_config_file)
    model = BigGAN(config)
    # pytorch_total_params = sum(p.numel() for p in model.parameters())
    # print(total_params)
    if load_tf:
        model = load_tf_weights_in_biggan(
            model,
            config,
            "./models/model_128/",
            "./models/model_128/batchnorms_stats.bin",
        )
        torch.save(model.state_dict(), cache_path)
    else:
        model.load_state_dict(torch.load(cache_path), strict=False)

    model.eval()

    # truncation = 0.4
    # noise = truncated_noise_sample(batch_size=2, truncation=truncation)
    # label = one_hot_from_names('diver', batch_size=2)

    # Prepare a input
    truncation = 0.4
    labels = one_hot_from_names(["soap bubble", "coffee", "mushroom"], batch_size=3)
    noises = truncated_noise_sample(truncation=truncation, batch_size=3)

    noises = torch.tensor(noises, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.float)
    with torch.no_grad():
        outputs, cond = model(noises, labels, truncation)

    import torchvision

    for index, (img, label, noise) in enumerate(zip(outputs, labels, noises)):
        class_num = torch.argmax(label)
        torchvision.utils.save_image(
            img * 0.5 + 0.5, "image256/{}/{}.png".format(class_num, index)
        )
        torch.save(noise, "image256/{}/{}.pt".format(class_num, index))
