# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
import matplotlib.pyplot as plt


from typing import List, Tuple, Type, Optional

from .common import LayerNorm2d

import pandas as pd

class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        image_encoder_type: str,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        image_size,
        patch_size,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        if image_encoder_type == 'swin_vit':
            self.feat_shape = image_size/patch_size
            self.output_upscaling = nn.Sequential(
                nn.ConvTranspose3d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                nn.LayerNorm((transformer_dim // 4, int(self.feat_shape[0]), int(self.feat_shape[1]), int(self.feat_shape[2]))),    # swin
                activation(),
                nn.ConvTranspose3d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),        # swin
                # nn.Conv3d(transformer_dim // 4, transformer_dim // 8, kernel_size=3, stride=1, padding=1),    # vit
                activation(),
            )
        else:
            self.feat_shape = image_size/patch_size * 2
            self.output_upscaling = nn.Sequential(
                nn.ConvTranspose3d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                nn.LayerNorm((transformer_dim // 4, int(self.feat_shape[0]), int(self.feat_shape[1]), int(self.feat_shape[2]))), # vit
                activation(),
                nn.ConvTranspose3d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                # nn.Conv3d(transformer_dim // 4, transformer_dim // 8, kernel_size=3, stride=1, padding=1),
                activation(),
            )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        self.txt_align_upscaled_embedding = nn.Linear(768, 96)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embedding: Optional[torch.Tensor],
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        # print('--------------decoder here--------------')
        masks = self.predict_masks(
            image_embeddings=image_embeddings,
            text_embedding=text_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :, :]
        # iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        text_embedding: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)    # [2, 7=(5+2), 256]
        # Expand per-image data in batch direction to be per-mask
        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w, d = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w, d)
        # print('src ', src.shape) # vit:[B, 768, 12, 12, 6], swin: [B, 6, 6, 3]
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w, d = upscaled_embedding.shape
        # print('hyper_in ', hyper_in.shape)    # [2, 4, 96]
        # print('upscaled_embedding ', upscaled_embedding.shape)    # [2, 96, 24, 24, 12]*
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w * d)).view(b, -1, h, w, d)
        # print('masks here ', masks.shape) # [2, 4, 24, 24, 12]
        # self.spatial_to_semantic(masks, upscaled_embedding.view(b, c, h * w * d), h, w, d)

        if text_embedding is not None:
            # text_embedding: B x 768, upscaled_embedding: B x c x h x w x d => B x 1 x h x w x d
            text_embedding_down = self.txt_align_upscaled_embedding(text_embedding).unsqueeze(dim=1)
            upscaled_embedding = upscaled_embedding.view(b, c, h * w * d)
            # print('text_embedding_down ', text_embedding_down.shape)  # [2, 1, 96]
            # text_embedding_norm = F.normalize(text_embedding_down, dim=-1)
            # upscaled_embedding_norm = F.normalize(upscaled_embedding, dim=1)
            # sim = (text_embedding_norm @ upscaled_embedding_norm).view(b, -1, h, w, d)
            # print(text_embedding_down.shape, upscaled_embedding.shape)
            sim = (text_embedding_down @ upscaled_embedding).view(b, -1, h, w, d)
            sim = sim.repeat(1, masks.shape[1], 1, 1, 1)
            masks = masks + sim
            # print('sim after', sim.shape) # [B, 4, 24, 24, 12]

            # ### ### ### ### ### 
            # masks_pixel = torch.where(torch.sigmoid(masks) > 0.5, 1.0, 0.0)

            # # flattened_data = (sim * masks_pixel).numpy().flatten()
            # # plt.hist(flattened_data, bins=50, color='blue', edgecolor='black')

            # # plt.title('Distribution of Flattened Tensor Values')
            # # plt.xlabel('Value')
            # # plt.ylabel('Frequency')

            # # plt.savefig('distribution_plot.png')
            
            
            # print('involved pixel n ', torch.sum(masks_pixel))
            # pixel_preds = torch.sigmoid(sim)
            # print('sim ', torch.unique(pixel_preds), torch.sum(pixel_preds))  # [B, 1, 24, 24, 12]
            # pixel_preds = torch.sigmoid(sim) * masks_pixel
            # print('masked sim ', torch.unique(pixel_preds), torch.sum(pixel_preds))  # [B, 1, 24, 24, 12]

            # masks = sim * masks_pixel
            ###### ### ### ### ### 

        return masks

    def spatial_to_semantic(self, masks, upscaled_embedding, h, w, d):
        # TODO
        ###
        text_embedding_pd = pd.read_pickle('text_embedding_down.pkl')
        foreground_mask = torch.where(torch.sigmoid(masks) > 0.5, 1.0, 0.0)
        sim_list, categories = [], []
        for index, row in text_embedding_pd.iterrows():
            category = row['Info']
            text_embedding = row['Tensor']

            sim = (text_embedding @ upscaled_embedding).view(h, w, d)
            # sim[sim<2.0] = 0.0
            sim_list.append(sim)
            categories.append(category)

            ##
            flattened_data = sim.flatten().numpy()
            plt.hist(flattened_data, bins=50, color='blue', edgecolor='black')

            plt.title('Distribution of Flattened Tensor Values')
            plt.xlabel('Value')
            plt.ylabel('Frequency')

            plt.savefig(f'distribution_{categories[index]}.png')
            plt.close()
            # sim[sim < 0.0] = 0.0
        ###
            
        sims = torch.stack(sim_list, dim=0)
        print(sims.shape)
        sims = F.softmax(sims, dim=0)
        
        
        for index in range(sims.shape[0]):
            forground_sim = sims[index] * foreground_mask
            flattened_data = forground_sim.flatten().numpy()
            plt.hist(flattened_data[flattened_data>0.0], bins=50, color='blue', edgecolor='black')

            plt.title('Distribution of Flattened Tensor Values')
            plt.xlabel('Value')
            plt.ylabel('Frequency')

            plt.savefig(f'distribution_softmax{categories[index]}.png')
            plt.close()
            print(categories[index], np.sum(flattened_data[flattened_data>0.0]))
        pass

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
