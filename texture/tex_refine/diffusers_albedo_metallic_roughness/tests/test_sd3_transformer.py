import sys
sys.path.append("/aigc_cfs_2/neoshang/code/diffusers_triplane")

from src.diffusers.models.transformers_model.transformer_vector import SD3Transformer2DModel

import torch
from thop import profile

device = torch.device("cuda:0")
"""
test cross attention
"""
# x = torch.randn([4, 512, 768])
# y = torch.randn([4, 197, 768])
# attention = Attention(dim=768, num_heads=8, qkv_bias=True)
# cross_attention = Attention(dim=768, num_heads=8, qkv_bias=True)
# # result = attention(x)
# result = cross_attention(x, y)
# print(result.shape)


"""
test position embedding
"""
# pos_embedding = get_2d_sincos_pos_embed(512, 16)
# pos_embed = get_1d_sincos_pos_embed_from_grid(768, np.array(range(512)))
# print(pos_embed.shape)

"""
test dit
"""
model = SD3Transformer2DModel(input_length_max = 2048,
                            in_channels = 8,
                            num_layers = 18,
                            attention_head_dim = 64,
                            num_attention_heads = 18,
                            joint_attention_dim = 4096,
                            caption_projection_dim = 1024,
                            pooled_projection_dim = 4096,
                            out_channels = 8).to(device)

x = torch.randn([2, 512, 8]).to(device)
condition = torch.randn([2, 1024, 1024]).to(device)
condition_pool = torch.randn([2, 4096]).to(device)
t = torch.randint(0, 1000, (2,), dtype=torch.long).to(device)

output = model(x, condition, condition_pool, t).sample
print(x.shape)
print(condition.shape)
print(condition_pool.shape)
print(t.shape)
print(output.shape)

Flops, params = profile(model, inputs=(x, condition, condition_pool, t)) # macs
print('Flops: % .4fG'%(Flops / 1000000000)) # 计算量
print('params: % .4fM'% (params / 1000000)) #参数量：等价与上面的summary输出的Total params值