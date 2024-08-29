import torch
import torch.nn as nn
from torch import einsum
import torchvision
from extern.guided_diffusion import utils
from einops import rearrange, repeat


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            utils.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, query_dim, out_dim, context_dim=1, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            # nn.Linear(inner_dim, query_dim),
            nn.Linear(inner_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context):
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)




class Model(nn.Module):
    def __init__(self, out_dim, cam_in, modelname: str = "dino_vits16", scale_factors: list = [1, 1 / 2, 1 / 3],
                 use_bn_in_head=False, norm_last_layer_in_head=True, modelcls='all'):
        super(Model, self).__init__()
        self.scale_factors = scale_factors
        self.modelcls = modelcls

        if "res" in modelname:
            self.backbone = getattr(torchvision.models, modelname)(pretrained=True)
            self.embed_dim = self.backbone.fc.weight.shape[1]
            self.backbone.fc = nn.Identity()
        elif "dinov2" in modelname:
            self.backbone = torch.hub.load("facebookresearch/dinov2", modelname)
            self.embed_dim = self.backbone.embed_dim
        elif "dino" in modelname:
            self.backbone = torch.hub.load("facebookresearch/dino:main", modelname)
            self.embed_dim = self.backbone.embed_dim
            self.backbone.head = nn.Identity()
        else:
            raise ValueError(f"Unknown model name {modelname}")

        for name, p in self.backbone.named_parameters():
            if name.find("block.10") >= 0 or name.find("block.11") >= 0 or name.find("block.9") >= 0:
                p.requires_grad = True
            else:
                p.requires_grad = False
            # p.requires_grad = False

        self.head = DINOHead(self.embed_dim * 2, out_dim, use_bn=use_bn_in_head,
                             norm_last_layer=norm_last_layer_in_head,)

        if self.modelcls in ['all', 'dir']:
            self.mapping_net = DINOHead(cam_in, self.embed_dim, use_bn=use_bn_in_head,
                                        norm_last_layer=norm_last_layer_in_head,
                                        nlayers=4, hidden_dim=256, bottleneck_dim=256)
            self.crossatt = CrossAttention(self.embed_dim, self.embed_dim * 2)
            self.norm = nn.LayerNorm(self.embed_dim * 2)

        # for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
        #     self.register_buffer(name, torch.FloatTensor(value).view(1, 3, 1, 1), persistent=False)



    def forward(self, x1, x2, cam):
        # x1/x2: 64 3 224 224
        batch_size = len(cam)
        assert batch_size == x2.shape[0] and batch_size == x1.shape[0]
        # assert batch_size == x2[0].shape[0] and batch_size == x1[0].shape[0]

        x1 = self.extract_dinofeat(x1) # bs x 384
        x2 = self.extract_dinofeat(x2)
        joint_feat = torch.cat((x1, x2), dim=1)

        if self.modelcls in ['all', 'dir']:
            cam = cam.view(batch_size, -1)
            # compute feature and camera attention
            cam_emb = self.mapping_net(cam)

            att_emb = self.crossatt(cam_emb.unsqueeze(1), joint_feat.unsqueeze(2))
            att_emb = self.norm(att_emb.squeeze()) + joint_feat
        else:
            att_emb = joint_feat

        pred = self.head(att_emb)
        return pred

    def extract_dinofeat(self, x):
        # x = self._resnet_normalize_image(x)
        features = self._compute_multiscale_features(x)
        return features

        # # convert to list
        # if not isinstance(x, list):
        #     x = [x]
        # idx_crops = torch.cumsum(torch.unique_consecutive(
        #     torch.tensor([inp.shape[-1] for inp in x]),
        #     return_counts=True,
        # )[1], 0)
        # start_idx, output = 0, torch.empty(0).to(x[0].device)
        # for end_idx in idx_crops:
        #     _out = self.backbone(torch.cat(x[start_idx: end_idx]))
        #     # The output is a tuple with XCiT model. See:
        #     # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
        #     if isinstance(_out, tuple):
        #         _out = _out[0]
        #     # accumulate outputs
        #     output = torch.cat((output, _out))
        #     start_idx = end_idx
        # # Run the head forward on the concatenated features.
        # return output

    # def _resnet_normalize_image(self, img: torch.Tensor) -> torch.Tensor:
    #     return (img - self._resnet_mean) / self._resnet_std

    def _compute_multiscale_features(self, img_normed: torch.Tensor) -> torch.Tensor:
        multiscale_features = None

        if len(self.scale_factors) <= 0:
            raise ValueError(f"Wrong format of self.scale_factors: {self.scale_factors}")

        for scale_factor in self.scale_factors:
            if scale_factor == 1:
                inp = img_normed
            else:
                inp = self._resize_image(img_normed, scale_factor)

            if multiscale_features is None:
                multiscale_features = self.backbone(inp)
            else:
                multiscale_features += self.backbone(inp)

        averaged_features = multiscale_features / len(self.scale_factors)
        return averaged_features

    @staticmethod
    def _resize_image(image: torch.Tensor, scale_factor: float) -> torch.Tensor:
        return nn.functional.interpolate(image, scale_factor=scale_factor, mode="bilinear", align_corners=False)





