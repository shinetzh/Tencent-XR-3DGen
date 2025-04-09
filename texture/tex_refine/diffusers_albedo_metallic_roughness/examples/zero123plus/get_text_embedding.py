import torch
import random
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer

device = torch.device("cuda")

model_pretrain_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/configs/zero123plus/zero123plus_v24_4views"
tokenizer = CLIPTokenizer.from_pretrained(
   model_pretrain_dir, subfolder="tokenizer"
)


text_encoder = CLIPTextModel.from_pretrained(
    model_pretrain_dir, subfolder="text_encoder"
)

def tokenize_captions(examples, is_train=True):
    captions = []
    for caption in examples:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                f"Caption column should contain either strings or lists of strings."
            )
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids

prompt_embedding = tokenize_captions([""]).to(device)
prompt_embedding_batch = torch.cat([prompt_embedding], dim=0)
text_encoder.to(device)
encoder_hidden_states_prompt = text_encoder(prompt_embedding_batch)[0]

torch.save(encoder_hidden_states_prompt.cpu(), "empty_prompt_embedding.pt")