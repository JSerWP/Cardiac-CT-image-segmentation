import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextConfig

#%% set up model
class SegVol(nn.Module):
    def __init__(self, 
                image_encoder, 
                mask_decoder,
                prompt_encoder,
                clip_ckpt,
                roi_size,
                patch_size,
                test_mode=False,
                ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.text_encoder = TextEncoder(clip_ckpt)
        self.feat_shape = np.array(roi_size)/np.array(patch_size)
        self.test_mode = test_mode

    def forward(self, image, text=None, boxes=None, points=None, **kwargs):
        bs = image.shape[0]
        img_shape = (image.shape[2], image.shape[3], image.shape[4])
        image_embedding, _ = self.image_encoder(image)
        image_embedding = image_embedding.transpose(1, 2).view(bs, -1, 
            int(self.feat_shape[0]), int(self.feat_shape[1]), int(self.feat_shape[2]))
        # test mode
        # predifined categories
        predifined_categories = ['liver', 'spleen', 'aorta', 'stomach', 'pancreas', 'right kidney', 'left kidney', 'tumor', 'right lung', 'left lung', 'heart', 'gallbladder', 'trachea']
        predifined_logits_list = []
        for cls in predifined_categories:
            cls_logits = self.forward_decoder(image_embedding, img_shape, [cls], None, None)
            predifined_logits_list.append(cls_logits)
        predifined_logits = torch.stack(predifined_logits_list, dim=0).squeeze()
        predifined_preds = F.softmax(predifined_logits, dim=0)
        predifined_res = torch.argmax(predifined_preds, dim=0)
        
        score = {}
        if self.test_mode:
            target_logits = self.forward_decoder(image_embedding, img_shape, text, boxes, points)
            return target_logits
            # foreground_mask = torch.where(torch.sigmoid(target_logits) > 0.5, True, False)
            # foreground_mask = foreground_mask.squeeze()
            # unique_values, cnts = torch.unique(predifined_res[foreground_mask], return_counts=True)
            # most_freq_value = unique_values[cnts.argmax()]
            # # print(f'the mask is {predifined_categories[most_freq_value]}')
            # score['res'] = predifined_categories[most_freq_value]
            # for idx, cls in enumerate(predifined_categories):
            #     predifined_preds_cls_foreground = torch.mean(predifined_preds[idx][foreground_mask])
            #     # print(f'{cls}-preds {predifined_preds_cls_foreground:.4f}')
            #     score[cls] = predifined_preds_cls_foreground
            # return target_logits, score
        # train mode
        # future release

    def forward_decoder(self, image_embedding, img_shape, text=None, boxes=None, points=None):
        with torch.no_grad():
            if boxes is not None:
                if len(boxes.shape) == 2:
                    boxes = boxes[:, None, :] # (B, 1, 6)
            if text is not None:
                text_embedding = self.text_encoder(text)  # (B, 768)
            else:
                text_embedding = None
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=None,
            text_embedding=text_embedding, # text_embedding
        )

        dense_pe = self.prompt_encoder.get_dense_pe()
        low_res_masks = self.mask_decoder(
            image_embeddings=image_embedding,
            text_embedding = text_embedding,
            image_pe=dense_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
          )
        logits = F.interpolate(low_res_masks, size=img_shape, mode='trilinear', align_corners=False)

        return logits

class TextEncoder(nn.Module):
    def __init__(self, clip_ckpt):
        super().__init__()
        config = CLIPTextConfig()
        self.clip_text_model = CLIPTextModel(config)
        self.tokenizer = AutoTokenizer.from_pretrained(clip_ckpt)
        self.dim_align = nn.Linear(512, 768)
        # freeze text encoder
        for param in self.clip_text_model.parameters():
            param.requires_grad = False

    def organ2tokens(self, organ_names):
        text_list = ['A computerized tomography of a {}.'.format(organ_name) for organ_name in organ_names]
        # text_list = ['a {}.'.format(organ_name) for organ_name in organ_names]
        tokens = self.tokenizer(text_list, padding=True, return_tensors="pt")
        return tokens
    
    def forward(self, text):
        if text is None:
            return None
        if type(text) is str:
            text = [text]
        tokens = self.organ2tokens(text)
        clip_outputs = self.clip_text_model(**tokens)
        text_embedding = clip_outputs.pooler_output
        text_embedding = self.dim_align(text_embedding)
        return text_embedding
