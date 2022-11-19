import torch
import torch.nn as nn




'''
BOS [SEQ] EOS PAD
其实就是自己预测自己
'''
class RNN(nn.Module):
    def __init__(self,src_length,embeding_dim):
        super(RNN,self).__init__()
        self.rnn = nn.LSTM(embeding_dim,embeding_dim,4)
        self.embeding = nn.Embedding(src_length,embeding_dim)
        self.fc = nn.Linear(embeding_dim,src_length)
    def forward(self,x):
        input_tensor = self.embeding(x)
        input_tensor = torch.transpose(input_tensor,0,1)
        input_tensor,_ = self.rnn(input_tensor)
        input_tensor = self.fc(input_tensor)
        out = nn.LogSoftmax(dim=-1)(input_tensor)
        return out


if __name__ == "__main__":
    model = RNN(5000,512)

    seq = torch.randint(0,5000,(6,10))

    output,(h,c) = model(seq)
    print(output.size())
    print(h.size())
    print(c.size())