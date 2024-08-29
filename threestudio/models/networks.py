import math

import tinycudann as tcnn
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.utils.base import Updateable
from threestudio.utils.config import config_to_primitive
from threestudio.utils.misc import get_rank
from threestudio.utils.ops import get_activation
from threestudio.utils.typing import *

from extern.ldm_zero123.modules.attention import FeedForward
from einops import rearrange, repeat
from torch import einsum


## new add
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, context_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if mask is not None:
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = torch.mul(attn.permute(0, 2, 1), v)

        # out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)

class LayerNorm(nn.LayerNorm):
    def forward(self, input: Tensor) -> Tensor:
        orig_type = input.dtype
        ret = super().forward(input.type(torch.float32))
        return ret.type(orig_type)


class ProgressiveBandFrequency(nn.Module, Updateable):
    def __init__(self, in_channels: int, config: dict):
        super().__init__()
        self.N_freqs = config["n_frequencies"]
        self.in_channels, self.n_input_dims = in_channels, in_channels
        self.funcs = [torch.sin, torch.cos]
        self.freq_bands = 2 ** torch.linspace(0, self.N_freqs - 1, self.N_freqs)
        self.n_output_dims = self.in_channels * (len(self.funcs) * self.N_freqs)
        self.n_masking_step = config.get("n_masking_step", 0)
        self.update_step(
            None, None
        )  # mask should be updated at the beginning each step

    def forward(self, x):
        out = []
        for freq, mask in zip(self.freq_bands, self.mask):
            for func in self.funcs:
                out += [func(freq * x) * mask]
        return torch.cat(out, -1)

    def update_step(self, epoch, global_step, on_load_weights=False):
        if self.n_masking_step <= 0 or global_step is None:
            self.mask = torch.ones(self.N_freqs, dtype=torch.float32)
        else:
            self.mask = (
                1.0
                - torch.cos(
                    math.pi
                    * (
                        global_step / self.n_masking_step * self.N_freqs
                        - torch.arange(0, self.N_freqs)
                    ).clamp(0, 1)
                )
            ) / 2.0
            threestudio.debug(
                f"Update mask: {global_step}/{self.n_masking_step} {self.mask}"
            )


class TCNNEncoding(nn.Module):
    def __init__(self, in_channels, config, dtype=torch.float32) -> None:
        super().__init__()
        self.n_input_dims = in_channels
        with torch.cuda.device(get_rank()):
            self.encoding = tcnn.Encoding(in_channels, config, dtype=dtype)
        self.n_output_dims = self.encoding.n_output_dims

    def forward(self, x):
        return self.encoding(x)


class ProgressiveBandHashGrid(nn.Module, Updateable):
    def __init__(self, in_channels, config, dtype=torch.float32):
        super().__init__()
        self.n_input_dims = in_channels
        encoding_config = config.copy()
        encoding_config["otype"] = "Grid"
        encoding_config["type"] = "Hash"
        with torch.cuda.device(get_rank()):
            self.encoding = tcnn.Encoding(in_channels, encoding_config, dtype=dtype)
        self.n_output_dims = self.encoding.n_output_dims
        self.n_level = config["n_levels"]
        self.n_features_per_level = config["n_features_per_level"]
        self.start_level, self.start_step, self.update_steps = (
            config["start_level"],
            config["start_step"],
            config["update_steps"],
        )
        self.current_level = self.start_level
        self.mask = torch.zeros(
            self.n_level * self.n_features_per_level,
            dtype=torch.float32,
            device=get_rank(),
        )

    def forward(self, x):
        enc = self.encoding(x)
        enc = enc * self.mask
        return enc

    def update_step(self, epoch, global_step, on_load_weights=False):
        current_level = min(
            self.start_level
            + max(global_step - self.start_step, 0) // self.update_steps,
            self.n_level,
        )
        if current_level > self.current_level:
            threestudio.debug(f"Update current level to {current_level}")
        self.current_level = current_level
        self.mask[: self.current_level * self.n_features_per_level] = 1.0


class CompositeEncoding(nn.Module, Updateable):
    def __init__(self, encoding, include_xyz=False, xyz_scale=2.0, xyz_offset=-1.0):
        super(CompositeEncoding, self).__init__()
        self.encoding = encoding
        self.include_xyz, self.xyz_scale, self.xyz_offset = (
            include_xyz,
            xyz_scale,
            xyz_offset,
        )
        self.n_output_dims = (
            int(self.include_xyz) * self.encoding.n_input_dims
            + self.encoding.n_output_dims
        )

    def forward(self, x, *args):
        return (
            self.encoding(x, *args)
            if not self.include_xyz
            else torch.cat(
                [x * self.xyz_scale + self.xyz_offset, self.encoding(x, *args)], dim=-1
            )
        )


def get_encoding(n_input_dims: int, config) -> nn.Module:
    # input suppose to be range [0, 1]
    encoding: nn.Module
    if config.otype == "ProgressiveBandFrequency":
        encoding = ProgressiveBandFrequency(n_input_dims, config_to_primitive(config))
    elif config.otype == "ProgressiveBandHashGrid":
        encoding = ProgressiveBandHashGrid(n_input_dims, config_to_primitive(config))
    else:
        encoding = TCNNEncoding(n_input_dims, config_to_primitive(config))
    encoding = CompositeEncoding(
        encoding,
        include_xyz=config.get("include_xyz", False),
        xyz_scale=2.0,
        xyz_offset=-1.0,
    )  # FIXME: hard coded
    return encoding


class VanillaMLP(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, config: dict):
        super().__init__()
        self.n_neurons, self.n_hidden_layers = (
            config["n_neurons"],
            config["n_hidden_layers"],
        )
        layers = [
            self.make_linear(dim_in, self.n_neurons, is_first=True, is_last=False),
            self.make_activation(),
        ]
        for i in range(self.n_hidden_layers - 1):
            layers += [
                self.make_linear(
                    self.n_neurons, self.n_neurons, is_first=False, is_last=False
                ),
                self.make_activation(),
            ]
        layers += [
            self.make_linear(self.n_neurons, dim_out, is_first=False, is_last=True)
        ]
        self.layers = nn.Sequential(*layers)
        self.output_activation = get_activation(config.get("output_activation", None))

    def forward(self, x):
        # disable autocast
        # strange that the parameters will have empty gradients if autocast is enabled in AMP
        with torch.cuda.amp.autocast(enabled=False):
            x = self.layers(x)
            x = self.output_activation(x)
        return x

    def make_linear(self, dim_in, dim_out, is_first, is_last):
        layer = nn.Linear(dim_in, dim_out, bias=False)
        return layer

    def make_activation(self):
        return nn.ReLU(inplace=True)


class SphereInitVanillaMLP(nn.Module):
    def __init__(self, dim_in, dim_out, config):
        super().__init__()
        self.n_neurons, self.n_hidden_layers = (
            config["n_neurons"],
            config["n_hidden_layers"],
        )
        self.sphere_init, self.weight_norm = True, True
        self.sphere_init_radius = config["sphere_init_radius"]
        self.sphere_init_inside_out = config["inside_out"]

        self.layers = [
            self.make_linear(dim_in, self.n_neurons, is_first=True, is_last=False),
            self.make_activation(),
        ]
        for i in range(self.n_hidden_layers - 1):
            self.layers += [
                self.make_linear(
                    self.n_neurons, self.n_neurons, is_first=False, is_last=False
                ),
                self.make_activation(),
            ]
        self.layers += [
            self.make_linear(self.n_neurons, dim_out, is_first=False, is_last=True)
        ]
        self.layers = nn.Sequential(*self.layers)
        self.output_activation = get_activation(config.get("output_activation", None))

    def forward(self, x):
        # disable autocast
        # strange that the parameters will have empty gradients if autocast is enabled in AMP
        with torch.cuda.amp.autocast(enabled=False):
            x = self.layers(x)
            x = self.output_activation(x)
        return x

    def make_linear(self, dim_in, dim_out, is_first, is_last):
        layer = nn.Linear(dim_in, dim_out, bias=True)

        if is_last:
            if not self.sphere_init_inside_out:
                torch.nn.init.constant_(layer.bias, -self.sphere_init_radius)
                torch.nn.init.normal_(
                    layer.weight,
                    mean=math.sqrt(math.pi) / math.sqrt(dim_in),
                    std=0.0001,
                )
            else:
                torch.nn.init.constant_(layer.bias, self.sphere_init_radius)
                torch.nn.init.normal_(
                    layer.weight,
                    mean=-math.sqrt(math.pi) / math.sqrt(dim_in),
                    std=0.0001,
                )
        elif is_first:
            torch.nn.init.constant_(layer.bias, 0.0)
            torch.nn.init.constant_(layer.weight[:, 3:], 0.0)
            torch.nn.init.normal_(
                layer.weight[:, :3], 0.0, math.sqrt(2) / math.sqrt(dim_out)
            )
        else:
            torch.nn.init.constant_(layer.bias, 0.0)
            torch.nn.init.normal_(layer.weight, 0.0, math.sqrt(2) / math.sqrt(dim_out))

        if self.weight_norm:
            layer = nn.utils.weight_norm(layer)
        return layer

    def make_activation(self):
        return nn.Softplus(beta=100)


class TCNNNetwork(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, config: dict) -> None:
        super().__init__()
        with torch.cuda.device(get_rank()):
            self.network = tcnn.Network(dim_in, dim_out, config)

    def forward(self, x):
        return self.network(x).float()  # transform to float32


class TextMLP(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, config: dict):
        super().__init__()
        self.n_neurons, self.n_hidden_layers = (
            config["n_neurons"],
            config["n_hidden_layers"],
        )
        # text
        text_dim_in = config["text_dim_in"]
        self.text_emb = None
        self.text_proj = self.make_linear(text_dim_in, dim_in, is_first=True, is_last=False)

        layers = [
            self.make_linear(dim_in * 2, self.n_neurons, is_first=True, is_last=False),
            self.make_activation(),
        ]
        for i in range(self.n_hidden_layers - 1):
            layers += [
                self.make_linear(
                    self.n_neurons, self.n_neurons, is_first=False, is_last=False
                ),
                self.make_activation(),
            ]
        layers += [
            self.make_linear(self.n_neurons, dim_out, is_first=False, is_last=True)
        ]
        self.layers = nn.Sequential(*layers)
        self.output_activation = get_activation(config.get("output_activation", None))

    def update_text_emb(self, text_emb):
        self.text_emb = text_emb.detach()

    def forward(self, x):
        if self.text_emb is None:
            raise ValueError("text_emb is None, please call update_text_emb first")
        # disable autocast
        # strange that the parameters will have empty gradients if autocast is enabled in AMP
        with torch.cuda.amp.autocast(enabled=False):
            text = self.text_proj(self.text_emb).repeat(x.size(0), 1)
            x = torch.cat((x, text), dim=-1)
            x = self.layers(x)
            x = self.output_activation(x)
        return x

    def make_linear(self, dim_in, dim_out, is_first, is_last):
        layer = nn.Linear(dim_in, dim_out, bias=False)
        return layer

    def make_activation(self):
        return nn.ReLU(inplace=True)

class AttentionMLP(TextMLP):
    def __init__(self, dim_in: int, dim_out: int, config: dict):
        super().__init__(dim_in, dim_out, config)

        text_dim_in = config["text_dim_in"]
        text_dim_out = config["n_hidden_text"]
        self.text_proj = self.make_linear(text_dim_in, text_dim_out, is_first=True, is_last=False)
        self.text_pos_attention = CrossAttention(text_dim_in, dim_in, dim_head=32, dropout=0.)
        self.ff = FeedForward(dim_in, dim_in, dropout=0., glu=True)
        self.norm1 = LayerNorm(dim_in)
        self.norm2 = LayerNorm(dim_in)

        layers = [
            self.make_linear(dim_in * 2 + text_dim_out, self.n_neurons, is_first=True, is_last=False),
            self.make_activation(),
        ]
        for i in range(self.n_hidden_layers - 1):
            layers += [
                self.make_linear(
                    self.n_neurons, self.n_neurons, is_first=False, is_last=False
                ),
                self.make_activation(),
            ]
        layers += [
            self.make_linear(self.n_neurons, dim_out, is_first=False, is_last=True)
        ]
        self.layers = nn.Sequential(*layers)
        self.output_activation = get_activation(config.get("output_activation", None))

    def forward(self, x):
        # disable autocast
        # strange that the parameters will have empty gradients if autocast is enabled in AMP
        if self.text_emb is None:
            raise ValueError("text_emb is None, please call update_text_emb first")
        with torch.cuda.amp.autocast(enabled=False):
            # calculate text attention, x:bsx32, text_emb:1x512
            x = x.unsqueeze(0)
            x_att = self.text_pos_attention(self.text_emb.unsqueeze(1), self.norm1(x)) + x
            x_att = self.ff(self.norm2(x_att)) + x_att
            x = torch.cat((x, x_att, self.text_proj(self.text_emb).unsqueeze(1).repeat(1, x.size(1), 1)), dim=-1).squeeze(0)

            x = self.layers(x)
            x = self.output_activation(x)
        return x


def get_mlp(n_input_dims, n_output_dims, config) -> nn.Module:
    network: nn.Module
    if config.otype == "VanillaMLP":
        network = VanillaMLP(n_input_dims, n_output_dims, config_to_primitive(config))
    elif config.otype == "SphereInitVanillaMLP":
        network = SphereInitVanillaMLP(
            n_input_dims, n_output_dims, config_to_primitive(config)
        )
    elif config.otype == "TextMLP":
        network = TextMLP(n_input_dims, n_output_dims, config_to_primitive(config))
    elif config.otype == "AttentionMLP":
        network = AttentionMLP(n_input_dims, n_output_dims, config_to_primitive(config))
    else:
        assert (
            config.get("sphere_init", False) is False
        ), "sphere_init=True only supported by VanillaMLP"
        network = TCNNNetwork(n_input_dims, n_output_dims, config_to_primitive(config))
    return network


class NetworkWithInputEncoding(nn.Module, Updateable):
    def __init__(self, encoding, network):
        super().__init__()
        self.encoding, self.network = encoding, network

    def forward(self, x):
        return self.network(self.encoding(x))


class TCNNNetworkWithInputEncoding(nn.Module):
    def __init__(
        self,
        n_input_dims: int,
        n_output_dims: int,
        encoding_config: dict,
        network_config: dict,
    ) -> None:
        super().__init__()
        with torch.cuda.device(get_rank()):
            self.network_with_input_encoding = tcnn.NetworkWithInputEncoding(
                n_input_dims=n_input_dims,
                n_output_dims=n_output_dims,
                encoding_config=encoding_config,
                network_config=network_config,
            )

    def forward(self, x):
        return self.network_with_input_encoding(x).float()  # transform to float32

def create_network_with_input_encoding(
    n_input_dims: int, n_output_dims: int, encoding_config, network_config
) -> nn.Module:
    # input suppose to be range [0, 1]
    network_with_input_encoding: nn.Module
    if encoding_config.otype in [
        "VanillaFrequency",
        "ProgressiveBandHashGrid",
    ] or network_config.otype in ["VanillaMLP", "SphereInitVanillaMLP", "TextMLP", "AttentionMLP"]:
        encoding = get_encoding(n_input_dims, encoding_config)
        network = get_mlp(encoding.n_output_dims, n_output_dims, network_config)
        network_with_input_encoding = NetworkWithInputEncoding(encoding, network)
    else:
        network_with_input_encoding = TCNNNetworkWithInputEncoding(
            n_input_dims=n_input_dims,
            n_output_dims=n_output_dims,
            encoding_config=config_to_primitive(encoding_config),
            network_config=config_to_primitive(network_config),
        )
    return network_with_input_encoding


class ToDTypeWrapper(nn.Module):
    def __init__(self, module: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return self.module(x).to(self.dtype)
