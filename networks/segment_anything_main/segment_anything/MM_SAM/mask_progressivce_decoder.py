# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Type
from .common import LayerNorm2d


class senet(nn.Module):
    def __init__(self, c=256, r=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(c, c // r, 1, 1, 0, bias=True), nn.ReLU(),
                                nn.Conv2d(c // r, c, 1, 1, 0, bias=True))
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

        self.apply(_init_weights)

    def forward(self, x):
        res = x
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        x = x * self.sigmoid(out)
        return x + res


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.act = nn.GELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.conv(x))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DeConv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.ConvTranspose2d(c1, c2, 2, 2, 0)
        # self.bn = nn.BatchNorm2d(c2)
        self.act = nn.GELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.conv(x))
        # return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=7, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s), g=c1)
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=c_)
        # self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x))  # if self.add else self.cv2(self.cv1(x))


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.se = senet(c=out_c)
        self.pw1 = Conv(in_c, out_c, 1, 1)
        self.pw2 = Conv(out_c, out_c, 1, 1)

    def forward(self, x):
        x = self.pw1(x)
        x = self.se(x)
        x = self.pw2(x)
        # x = self.se(x)
        return x


class conv_block1(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.pw1 = Conv(in_c, out_c, 1, 1)
        self.pw2 = Conv(out_c, out_c, 1, 1)

    def forward(self, x):
        x = self.pw1(x)
        x = self.pw2(x)
        return x


class conv_block_plus(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # self.se = senet(c = out_c)
        self.pw1 = Conv(in_c, out_c, 1, 1)
        self.axis_dw1 = CrossConv(c1=out_c, c2=out_c, k=7)
        self.axis_dw2 = CrossConv(c1=out_c, c2=out_c, k=3)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(out_c, out_c // 16, 1, 1, 0, bias=True), nn.ReLU(),
                                nn.Conv2d(out_c // 16, out_c, 1, 1, 0, bias=True))
        self.pw2 = Conv(out_c, out_c, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pw1(x)
        x1 = self.axis_dw1(x)
        x2 = self.axis_dw2(x)

        max_out = self.fc(self.max_pool(x1 + x2))
        x1 = x1 * self.sigmoid(max_out)
        x2 = x2 * self.sigmoid(max_out)

        # y = torch.cat([x1,x2],dim=1)
        y = x1 + x2
        res = self.pw2(y)

        return res


class conv_up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # self.se = senet(c = out_c)
        self.conv_up = nn.ConvTranspose2d(in_c, out_c, 2, 2, 0)
        self.conv_fu = Conv(out_c, out_c)

    def forward(self, x):
        x = self.conv_up(x)
        x = self.conv_fu(x)
        return x


class conv_up0(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # self.se = senet(c = out_c)
        self.conv_up = nn.ConvTranspose2d(in_c, out_c, 1, 1, 0)
        self.conv_fu = Conv(out_c, out_c)

    def forward(self, x):
        x = self.conv_up(x)
        x = self.conv_fu(x)
        return x


class conv_up_plus(nn.Module):
    def __init__(self, in_c, out_c, k=2, s=2):
        super().__init__()
        self.conv_down = nn.ConvTranspose2d(in_c, out_c, k, s, 0)
        # self.conv_fu = Conv(out_c,out_c)

    def forward(self, x):
        x = self.conv_down(x)
        # x = self.conv_fu(x)
        return x


class conv_down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # self.se = senet(c = out_c)
        self.conv_down = nn.MaxPool2d(2, 2)
        # self.conv_fu = Conv(in_c,out_c)

    def forward(self, x):
        x = self.conv_down(x)
        # x = self.conv_fu(x)
        return x


class conv_pre(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.pre = nn.Conv2d(in_c, 2, 1, 1, 0)

    def forward(self, x):
        x = self.pre(x)
        return x


class conv_up_pre(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # self.se = senet(c = out_c)
        self.up = Conv(in_c, out_c)
        self.pre = nn.Conv2d(out_c, 2, 1, 1, 0)
        # self.conv = CrossConv(in_c,out_c)

    def forward(self, x):
        x = self.up(x)
        x = self.pre(x)
        return x


class desam(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.de12_ = conv_block(768, 256)
        self.hyper = conv_block(256 * 2, 256)

    def forward(self, mask_in, mask_embed):
        de12_ = self.de12_(mask_embed.permute(0, 3, 1, 2))
        hyper = torch.cat([mask_in, de12_], dim=1)
        hyper = self.hyper(hyper)
        return mask_in + hyper


class MaskDecoder(nn.Module):
    def __init__(
            self,
            *,
            args,
            transformer_dim: int,
            transformer: nn.Module,
            num_prototypes=16,
            num_multimask_outputs: int = 2,
            activation: Type[nn.Module] = nn.GELU,
            iou_head_depth: int = 3,
            iou_head_hidden_dim: int = 224,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer           (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super(MaskDecoder, self).__init__()
        self.transformer_dim = transformer_dim
        self.transformer_prototype1 = transformer
        self.transformer_prototype2 = transformer
        self.transformer_prototype3 = transformer
        self.transformer_prototype4 = transformer
        self.args = args
        self.num_multimask_outputs = 2
        self.num_prototypes = num_prototypes
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = 2  # num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.prototype_tokens = nn.Embedding(self.num_prototypes, transformer_dim)
        self.prototype_mlp = nn.Linear(num_prototypes, self.num_multimask_outputs)

        self.output_upscaling_prototype = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 2, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 2),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 2, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 8),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 8, transformer_dim // 8, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 8),
            activation(),
        )
        self.output_hypernetworks_mlps_prototype1 = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim, 3)
                for i in range(self.num_prototypes + 1)
            ]
        )
        self.output_hypernetworks_mlps_prototype2 = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim, 3)
                for i in range(self.num_prototypes + 1)
            ]
        )
        self.output_hypernetworks_mlps_prototype3 = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim, 3)
                for i in range(self.num_prototypes + 1)
            ]
        )
        self.output_hypernetworks_mlps_prototype = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_prototypes + 1)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )
        self.desam1 = desam()
        self.desam2 = desam()
        self.desam3 = desam()
        self.desam4 = desam()

    # Prepare output
    # by LBK EDIT
    @staticmethod
    def interpolate(x, w, h):
        height, width = x.shape[2:]
        w0, h0 = w + 0.1, h + 0.1
        x = nn.functional.interpolate(
            x,
            scale_factor=(w0 / height, h0 / width),
            mode='bicubic',
        )
        return x

    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred, aux_loss = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )
        return masks, iou_pred, aux_loss

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
    ):
        """Predicts masks. See 'forward' for more details."""
        de_in = image_embeddings[1:-1]
        aux_loss = image_embeddings[-1]
        image_embeddings = image_embeddings[0]

        # Concatenate output tokens
        prototype_tokens = torch.cat([self.iou_token.weight, self.prototype_tokens.weight], dim=0)
        prototype_tokens = prototype_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        prototype_tokens = torch.cat((prototype_tokens, sparse_prompt_embeddings), dim=1)

        src = torch.repeat_interleave(image_embeddings, prototype_tokens.shape[0], dim=0)
        try:
            src = src + dense_prompt_embeddings
        except:
            src = src + self.interpolate(dense_prompt_embeddings, *src.shape[2:])

        pos_src1 = torch.repeat_interleave(image_pe, prototype_tokens.shape[0], dim=0)
        pos_src2 = torch.repeat_interleave(image_pe, prototype_tokens.shape[0], dim=0)
        pos_src3 = torch.repeat_interleave(image_pe, prototype_tokens.shape[0], dim=0)
        pos_src4 = torch.repeat_interleave(image_pe, prototype_tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        try:
            hs1, src1 = self.transformer_prototype1(src, pos_src1, prototype_tokens)
        except:
            hs1, src1 = self.transformer_prototype1(src, self.interpolate(pos_src1, *src.shape[2:]), prototype_tokens)
        src1 = src1.transpose(1, 2).view(b, c, h, w)
        src2 = self.desam1(src1, de_in[0])

        hyper_in_list_pro1 = []
        for i in range(self.num_prototypes + 1):
            hyper_in_list_pro1.append(self.output_hypernetworks_mlps_prototype1[i](hs1[:, i, :]))
        hs2 = hs1 + torch.stack(hyper_in_list_pro1, dim=1)

        try:
            hs2, src2 = self.transformer_prototype2(src2, pos_src2, hs2)
        except:
            hs2, src2 = self.transformer_prototype2(src2, self.interpolate(pos_src2, *src2.shape[2:]), hs2)
        src2 = src2.transpose(1, 2).view(b, c, h, w)
        src3 = self.desam2(src2, de_in[1])

        hyper_in_list_pro2 = []
        for i in range(self.num_prototypes + 1):
            hyper_in_list_pro2.append(self.output_hypernetworks_mlps_prototype2[i](hs2[:, i, :]))
        hs3 = hs2 + torch.stack(hyper_in_list_pro2, dim=1)

        try:
            hs3, src3 = self.transformer_prototype3(src3, pos_src3, hs3)
        except:
            hs3, src3 = self.transformer_prototype3(src3, self.interpolate(pos_src3, *src3.shape[2:]), hs3)
        src3 = src3.transpose(1, 2).view(b, c, h, w)
        src4 = self.desam3(src3, de_in[2])

        hyper_in_list_pro3 = []
        for i in range(self.num_prototypes + 1):
            hyper_in_list_pro3.append(self.output_hypernetworks_mlps_prototype3[i](hs3[:, i, :]))
        hs4 = hs3 + torch.stack(hyper_in_list_pro3, dim=1)

        try:
            hs4, src4 = self.transformer_prototype4(src4, pos_src4, hs4)
        except:
            hs4, src4 = self.transformer_prototype4(src4, self.interpolate(pos_src4, *src3.shape[2:]), hs4)
        src4 = src4.transpose(1, 2).view(b, c, h, w)
        src5 = self.desam4(src4, de_in[3])

        upscaled_embedding_prototype = self.output_upscaling_prototype(src5)
        iou_token_out = hs4[:, 0, :]

        prototype_tokens_out = hs4[:, 1: (1 + self.num_prototypes), :]

        hyper_in_list_pro = []
        for i in range(self.num_prototypes):
            hyper_in_list_pro.append(self.output_hypernetworks_mlps_prototype[i](prototype_tokens_out[:, i, :]))

        hyper_in_pro = torch.stack(hyper_in_list_pro, dim=1)
        b, c, h, w = upscaled_embedding_prototype.shape
        prototypes = (hyper_in_pro @ upscaled_embedding_prototype.view(b, c, h * w)).view(b, -1, h, w)

        mask_prototype = self.prototype_mlp(prototypes.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        iou_pred = self.iou_prediction_head(iou_token_out)
        return mask_prototype, iou_pred, aux_loss


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


