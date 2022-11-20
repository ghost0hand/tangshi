import torch
import torch.nn as nn
from net.transformer import Transformer,get_submask
from Dataset.TDataset import GetLoader
import setting
import json
import zhconv


# 使用transformer写唐诗
class RnnWriteTangshi():
    def __init__(self,key_path,model_path):

        with open(key_path, "r", encoding="utf-8") as f:
            self.index_code, self.code_index, self.length = json.load(f)
        # 加载模型
        self.model = Transformer(512,4,self.length,self.length).to(setting.DEVICE)
        param = torch.load(model_path)
        self.model.load_state_dict(param)
        self.model.eval()

    def Trans(self,wrods):
        out = [self.code_index["BOS"]]
        for code in wrods:
            out.append(self.code_index[code])
        out.append(self.code_index["EOS"])
        return out

    def Write(self,timu):
        timu = zhconv.convert(timu,"zh-hk")
        timu_ = self.Trans(timu)
        src = torch.tensor([timu_],dtype=torch.int64)
        x = self.model.Encode(src)
        seq = [self.code_index["BOS"]]
        tager = torch.tensor(seq,dtype = torch.int64).to(setting.DEVICE)
        tager = torch.unsqueeze(tager,dim = 0)
        for i in range(150):
            mask = get_submask(i + 1).to(setting.DEVICE)
            tager = self.model.Decode(x,tager,mask = mask)
            tager = nn.LogSoftmax(dim = -1)(self.model.linear(tager))
            _,pred = torch.max(tager,dim = 2)
            data = pred[-1,-1]
            seq.append(data.item())
            tager = torch.tensor(seq,dtype = torch.int64).to(setting.DEVICE)
            tager = torch.unsqueeze(tager,dim = 0)
            if data.item() == self.code_index["EOS"]:
                break

        words = ''
        for data in seq:
            word = self.index_code[str(data)]
            if word == "EOS":
                break
            if word == "BOS":
                words += " "
            else:
                words += word
        return timu,words

writer = RnnWriteTangshi(
    r'Dataset/key_.json',"model/model99.pht")
while True:
    timu = input("请输入题目:")
    timu,content = writer.Write(timu.strip())
    print("题目\n",timu)
    print("作诗\n",zhconv.convert(content,'zh-cn'))
    print("\n\n\n")
