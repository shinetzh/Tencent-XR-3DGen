import numpy as np
import torch
import matplotlib.pyplot as plt
from diffusers.optimization import get_scheduler

# Initialize the optimizer
optimizer_cls = torch.optim.AdamW

atensor = torch.randn([3, 256, 3, 3], requires_grad=True)

optimizer = optimizer_cls(
    [atensor],
    lr=7e-5,
    betas=(0.9, 0.999),
    weight_decay=1e-2,
    eps=1e-8,
)


lr_scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=48000,
    num_training_steps=3672054,
)

global_step = 0
lr_list = []
for epoch in range(0, 6):
    loss_list = []
    train_loss = 0.0
    for step in range(612008):
        optimizer.step(0)
        lr_scheduler.step()
        # print(lr_scheduler.get_last_lr()[0])
        global_step += 1
        lr_list.append(lr_scheduler.get_last_lr()[0])
breakpoint()
x = np.linspace(0, 3672048, 3672048)   # 定义域

plt.plot(x, lr_list, "g", linewidth=2)    # 加载曲线
plt.grid(True)  # 网格线
# plt.show()  # 显示

plt.savefig("lr_list.png")