import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import setting


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 初始化α为全1, 而β为全0
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # 按照最后一个维度计算均值和方差 ，embed dim
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x-mean) / (torch.sqrt(std**2 + self.eps)) + self.b_2

#多头注意力机制
class ScaledDotProductAttention(nn.Module):
    def __init__(self,temperature,att_dropout = 0.1):
        super(ScaledDotProductAttention,self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(att_dropout)

    def forward(self,q,k,mask = None):
        att = torch.matmul(q , k.transpose(2,3)) * self.temperature# Q * K
        if mask is not None:
            att = att.masked_fill(mask == 0,-1e9)
        att = self.dropout(F.softmax(att,dim = -1))
        return att

class MultHeadSelfAttention(nn.Module):
    def __init__(self,dim_in = 512,dim_k = 64,num_heads = 32,mode = 'en'):
        super(MultHeadSelfAttention,self).__init__()
        self.mode = mode
        self.dim_k = dim_k
        self.dim_in = dim_in
        self.num_heads = num_heads
        
        self.linear_q = nn.Linear(dim_in,dim_k * num_heads,bias = False)
        self.linear_k = nn.Linear(dim_in,dim_k * num_heads,bias = False)
        self.linear_v = nn.Linear(dim_in,dim_k * num_heads,bias = False)
        
        self.linear_z = nn.Linear(self.dim_k * self.num_heads,dim_in)

        self._norm_fact = 1 / math.sqrt(dim_k)

        self.decode_attention = ScaledDotProductAttention(temperature = self._norm_fact)

    def reshape(self,tensor):
        a = []
        for tensor in tensor.split(self.dim_k,dim = 2):
            a.append(torch.unsqueeze(tensor,dim = 1))
        return torch.cat(a,dim = 1)

    def forward(self,x,xq = None,mesk = None):
        batch , n , dim_in = x.size()
        assert dim_in == self.dim_in

        if self.mode == 'en' or self.mode == "de_m":
            q = self.reshape(self.linear_q(x))
            k = self.reshape(self.linear_k(x))
            v = self.reshape(self.linear_v(x))
        elif self.mode == 'de':
            q = self.reshape(self.linear_q(xq))
            k = self.reshape(self.linear_k(x))
            v = self.reshape(self.linear_v(x))
        if self.mode == 'en' or self.mode == "de":
            dist = torch.matmul(q,k.transpose(2,3)) * self._norm_fact
            dist = torch.softmax(dist,dim = -1)
            att = torch.matmul(dist,v)
            att = att.transpose(1,2)
            att = att.reshape(batch,att.size(1),self.dim_k * self.num_heads)
            att = self.linear_z(att)
            return att
        elif self.mode == "de_m":
            att = self.decode_attention(q,k,mesk)
            # print(att[0,0,:,:])
            att = torch.matmul(att,v)
            att = att.transpose(1,2)
            att = att.reshape(batch,n,self.dim_k * self.num_heads)
            att = self.linear_z(att)
            return att
#位置编码
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len = 5000):
        super(PositionalEncoding,self).__init__()

        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0,max_len).unsqueeze(dim = 1)
        div_term = 10000.0 ** -(torch.arange(0,d_model,2) / d_model)
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(dim = 0)
        self.register_buffer("pe",pe)

    def forward(self,x):
        return x + torch.tensor(self.pe[:,0:x.size(1),0:x.size(2)].clone().detach_(),requires_grad = False)


# 产生mash,输入句子最大的长度
def get_submask(max_len):
    tensor = torch.ones([max_len,max_len])
    tensor = torch.tril(tensor).bool()
    return tensor

#输入需要embedding的维度，句子最大长度，tensor
class Embedding(nn.Module):
    def __init__(self,src_length,out_dim):
        super(Embedding,self).__init__()
        weight = torch.randn(src_length,out_dim,requires_grad = True)
        self.weight = nn.Parameter(weight,requires_grad=True)
    def forward(self,x):
        assert (x.dtype == torch.int64 
            or x.dtype == torch.int32),"使用64或者32的int"
        out_tensor_list = []
        for tensor_item in x:
            out_tensor_list.append(torch.unsqueeze(self.weight[tensor_item],dim = 0))

        return torch.cat(out_tensor_list)
class Embeddings(nn.Module):
    def __init__(self,d_model,words_len):
        super(Embeddings,self).__init__()
        self.embed = Embedding(words_len,d_model)
    def forward(self,x):
        x = torch.tensor(x,dtype=torch.int64)
        return self.embed(x)

class FFN(nn.Module):
    def __init__(self,d_model):
        super(FFN,self).__init__()
        layer = []
        layer.append(nn.Linear(d_model,int(d_model * 2)))
        layer.append(nn.ReLU())
        layer.append(nn.Linear(int(d_model * 2),d_model))
        self.norm = LayerNorm(d_model)
        self.ffn = nn.Sequential(*layer)
    def forward(self,x):
        return self.norm(x + self.ffn(x))

class Encoder(nn.Module):
    def __init__(self,d_model):
        super(Encoder,self).__init__()
        self.attent = MultHeadSelfAttention()
        self.norm = LayerNorm(d_model)
    def forward(self,x):
        return self.norm(x + self.attent(x))

class EndederLayer(nn.Module):
    def __init__(self,N,d_model):
        super(EndederLayer,self).__init__()
        layer = []
        for _ in range(N):
            layer.append(Encoder(d_model))
            layer.append(FFN(d_model=d_model))
        self.decode = nn.Sequential(*layer)
        self.num = N
    def forward(self,x):
        for layer in range(self.num):
            x = self.decode(x)
        return x

class Decoder(nn.Module):
    def __init__(self,d_model,mode = 'de_m'):
        super(Decoder,self).__init__()
        self.mode = mode
        if mode == 'de_m':
            self.attent = MultHeadSelfAttention(mode=mode)
        elif mode == 'de':
            self.attent = MultHeadSelfAttention(mode=mode)
        self.norm = LayerNorm(d_model)
    def forward(self,x,xq = None,mask = None):
        if self.mode == 'de':
            return self.norm(xq + self.attent(x,xq = xq))
        elif self.mode == 'de_m':
            return self.norm(x + self.attent(x,mesk = mask))

class DedcoderLayer(nn.Module):
    def __init__(self,N,d_model):
        super(DedcoderLayer,self).__init__()
        layer = []
        self.num = N
        for _ in range(N):
            layer.append(Decoder(d_model))
            layer.append(Decoder(d_model,mode = 'de'))
            layer.append(FFN(d_model=d_model))

        self.layer = layer
        self.model = nn.Sequential(*layer)
    def forward(self,x_src,xq,mask = None):
        for index in range(self.num):
            x = self.layer[index * 3](xq,mask = mask)
            x = self.layer[index * 3 + 1](x_src,x)
            x = self.layer[index * 3 + 2](x)
        return x


class Transformer(nn.Module):
    def __init__(self,d_model,N,words_len_src,word_len_tager):
        super(Transformer,self).__init__()
        self.d_model = d_model

        self.words_len_src = words_len_src
        self.word_len_tager = word_len_tager

        #编码器
        self.encode = EndederLayer(N,d_model)

        #解码器
        self.decode = DedcoderLayer(N,d_model)

        #位置编码
        self.posencode = PositionalEncoding(d_model=d_model)

        #embdeeings
        self.embd_src = Embeddings(d_model=d_model,words_len=self.words_len_src)
        self.embd_tag = Embeddings(d_model=d_model,words_len=self.word_len_tager)

        #转换矩阵
        self.linear = nn.Linear(d_model,word_len_tager,bias = False)

    def Encode(self,x_src):
        x_src = torch.tensor(x_src,dtype=torch.int64)
        return self.encode(self.posencode(self.embd_src(x_src)))

    def Decode(self,x_src,x_tager,mask):
        x_tager = torch.tensor(x_tager,dtype=torch.int64)
        return self.decode(x_src,
            self.posencode(
                self.embd_tag(x_tager)
                ),
            mask = mask)

    def forward(self,x_src,x_tager):
        
        x_att = self.Encode(x_src)
        mask = get_submask(x_tager.size(1)).to(setting.DEVICE)
        x = self.Decode(x_att,x_tager,mask = mask)

        x = self.linear(x)
        x = nn.LogSoftmax(dim = -1)(x)
        return x


# batch,num,num_words
def Transformerloss(tensor_output,tensor_tager):
    loss = torch.nn.NLLLoss()
    tensor_tager = tensor_tager.view([tensor_tager.size()[0] * tensor_tager.size()[1]])
    tensor_output = tensor_output.contiguous().view([tensor_output.size()[0] * tensor_output.size()[1],tensor_output.size()[2]])
    tensor_tager = torch.tensor(tensor_tager,dtype = torch.int64)
    
    max_tensor = torch.max(tensor_output,dim = -1)
    # print(max_tensor[1])
    # print(tensor_tager)

    return loss(tensor_output,tensor_tager)


if __name__ == "__main__":
    x_src = torch.range(0,22,dtype = torch.int32).reshape(5,8)
    x_tager = torch.range(0,33,dtype = torch.int32).reshape(5,11)

    model = Transformer(512,8,8,11).to(setting.DEVICE)
    print(model(x_src,x_tager).size())
