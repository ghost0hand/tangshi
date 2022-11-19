from re import X
import torch
import torch.nn as nn
from net.transformer import Transformer,Transformerloss
from Dataset.TDataset import GetLoader
import os
import setting
import json
import tqdm



#参数
lr = 1e-6
loader = GetLoader()

with open(r'Dataset/key_.json',"r",encoding='utf-8') as f:
    index_code,code_index,length = json.load(f)

#网络
model = Transformer(512,2,length,length).to(setting.DEVICE)

# param = torch.load("model/model1.pht")
# model.load_state_dict(param)

#优化器
optimizer = torch.optim.Adam(model.parameters(),lr = lr,weight_decay=5e-4)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.8)

model_path = 'model'



if __name__ == "__main__":
    model.train()

    for epoch in range(0,100):
        tq = tqdm.tqdm(loader,total=len(loader),colour="blue")
        for i,batch in enumerate(tq):
            src = torch.tensor(batch['title'],dtype = torch.int32).to(setting.DEVICE)
            tager = torch.tensor(batch['content'],dtype = torch.int32).to(setting.DEVICE)
            label = torch.tensor(batch['label'],dtype = torch.int32).to(setting.DEVICE)

            words = ''
            for data in src[0]:
                word = index_code[str(data.item())]
                if word == "EOS":
                    break
                words += (word + " ")
            src_words = words
            words = ''
            for data in tager[0]:
                word = index_code[str(data.item())]
                if word == "EOS":
                    break
                words += word
                    
            tag_words = words
            
            x = model(src,tager)

            max_tensor = torch.max(x,dim = -1)
            acc = torch.sum(torch.eq(max_tensor[1],label)).item()
            all_num = label.size()[0] * label.size()[1]

            optimizer.zero_grad()
            loss = Transformerloss(x,label) * 10.0


            loss.backward()
            optimizer.step()
            tq.set_postfix(
                {
                    "acc/all_num":acc / all_num,
                    "loss:":loss.item(),
                    "epoch":epoch,
                    "lr":optimizer.param_groups[0]['lr']
                }
            )

        lr_scheduler.step()
        torch.save(model.state_dict(),os.path.join(model_path,"model%d.pth"%epoch))