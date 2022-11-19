import torch
import torch.nn as nn
import zhconv
from net.transformer import Transformer, Transformerloss
from Dataset.TDataset import GetLoader,GetLoaderRnn
import setting
import json
from net.rnn import RNN
import re



# 使用RNN写唐诗
class RnnWriteTangshi():
    def __init__(self,key_path,model_path):

        with open(key_path, "r", encoding="utf-8") as f:
            self.index_code, self.code_index, self.length = json.load(f)
        # 加载模型
        self.model = RNN(self.length, 512).to(setting.DEVICE)
        param = torch.load(model_path)
        self.model.load_state_dict(param)
        self.model.eval()
        self.matcomp = re.compile(r'BOS(.*?)EOSBOS(.*?)EOS')

    def Trans(self,wrods):
        out = [self.code_index["BOS"]]
        for code in wrods:
            out.append(self.code_index[code])
        out.append(self.code_index["EOS"])
        return out

    def Write(self,timu):
        timu = zhconv.convert(timu,"zh-hk")
        content = self.Trans(timu)
        while True:
            input_tensor = torch.tensor([content],dtype=torch.int64).to(setting.DEVICE)
            x = self.model(input_tensor)
            _,pred = torch.max(x,-1)

            pred_item = pred[-1,0].item()
            content.append(pred_item)
            if pred_item == self.code_index["EOS"]:
                break

        words = ""
        for item in content:
            words += self.index_code[str(item)]
        words = zhconv.convert(words,"zh-cn")

        return re.findall(self.matcomp,words)[0]


writer = RnnWriteTangshi(
    r'Dataset/keyRNN_.json',"modelrnn/model16.pth")
while True:
    timu = input("请输入题目:")
    timu,content = writer.Write(timu.strip())
    print("题目\n",timu)
    print("作诗\n",zhconv.convert(content,'zh-cn'))
    print("\n\n\n")
