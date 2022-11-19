import torch
import torch.nn as nn
from net.transformer import Transformerloss
from Dataset.TDataset import GetLoaderRnn
import os
import setting
import json
import tqdm
from net.rnn import RNN

# 参数
lr = 1e-3
loader = GetLoaderRnn()

with open(r'Dataset/key_.json', "r", encoding="utf-8") as f:
    index_code, code_index, length = json.load(f)

# 网络
model = RNN(length, 512).to(setting.DEVICE)


param = torch.load("modelrnn/model25.pth")
model.load_state_dict(param)


# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

model_path = 'modelrnn'

model.train()
for epoch in range(0, 100):

    tq = tqdm.tqdm(loader, total=len(loader))
    for i, batch in enumerate(tq):
        content = torch.tensor(batch['content'], dtype=torch.int32).to(setting.DEVICE)
        label = torch.tensor(batch['label'], dtype=torch.int32).to(setting.DEVICE)

        x = model(content)
        optimizer.zero_grad()
        x = torch.transpose(x,0,1)
        loss = Transformerloss(x, label) * 10.0

        words = ""
        _, pred = torch.max(x, -1)
        for item in pred[0]:
            if index_code[str(item.item())] != "PAD" and index_code[str(item.item())] != "EOS":
                words += index_code[str(item.item())]

        print(words)
        loss.backward()
        optimizer.step()
        tq.set_postfix(
            {
                "loss:": loss.item(),
                "epoch": epoch,
                "lr": optimizer.param_groups[0]['lr']
            }
        )

    lr_scheduler.step()
    torch.save(model.state_dict(), os.path.join(model_path, "model%d.pth" % epoch))