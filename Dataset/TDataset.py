import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import json

try:
    from Dataset.dataprocess import readFIels, processOneFile
except:
    from dataprocess import readFIels, processOneFile
import os


class TSdataset(Dataset):
    def __init__(self, file, key_path):
        files = readFIels(file)
        all_tangshi = []
        for file in files:
            all_tangshi += processOneFile(file)
        self.tangshi = all_tangshi

        with open(key_path, "r", encoding='utf-8') as f:
            self.index_code, self.code_index, self.length = json.load(f)

    def __getitem__(self, index):
        itemtangshi = self.tangshi[index]
        auth = itemtangshi["auth"]
        content = itemtangshi["content"]
        title = itemtangshi["title"]

        auth_ = []
        for code in auth:
            auth_.append(self.code_index[code])

        content_ = [self.code_index["BOS"]]
        for line in content:
            for code in line:
                content_.append(self.code_index[code])
        content_.append(self.code_index["EOS"])

        title_ = [self.code_index["BOS"]]
        for code in title:
            title_.append(self.code_index[code])
        title_.append(self.code_index["EOS"])

        return {
            "auth": auth_,
            "content": content_,
            "title": title_
        }

    def __len__(self):
        return len(self.tangshi)

def collate_fn(batch):
    """
        直接将各个语句进行拼接
        比如 题目 AAA 诗句 BBBBBBBBBB
        然后结果就是BOS AAA EOS 和 BOS BBBBBBBBBB EOS
        然后标签就是               BBBBBBBBBB EOS PAD
        正好是一个对应关系
    Args:
        batch (dict): dataset处理的内容

    Returns:
        _type_: 返回训练的句子和标签
    """
    del_save = []
    for i in range(len(batch)):
        if len(batch[i]["content"]) > 150:
            del_save.append(i)
    if len(del_save) > 0:
        for i in range(len(del_save) - 1, -1, -1):
            del batch[del_save[i]]

    max_length_title = 0
    max_length_auth = 0
    max_length_content = 0
    for item in batch:
        content = item["content"]
        auth = item["auth"]
        title = item["title"]

        if len(title) > max_length_title:
            max_length_title = len(title)
        if len(auth) > max_length_auth:
            max_length_auth = len(auth)
        if len(content) > max_length_content:
            max_length_content = len(content)
    title_ = []
    auth_ = []
    content_ = []

    for item in batch:
        title_.append(item["title"] + [2] * (max_length_title - len(item['title'])))
        auth_.append(item["auth"] + [2] * (max_length_auth - len(item['auth'])))
        content_.append(item["content"] + [2] * (max_length_content - len(item['content'])))

    label = content_.copy()
    content_ = torch.tensor(content_, dtype=torch.int64)
    for i in range(len(label)):
        del label[i][0]
        label[i].append(2)
    return {
        "auth": torch.tensor(auth_, dtype=torch.int64),
        "content": content_,
        "title": torch.tensor(title_, dtype=torch.int64),
        "label": label
    }

def collate_fn_rnn(batch):
    """
        直接将各个语句进行拼接
        比如 题目 AAA 诗句 BBBBBBBBBB
        然后结果就是BOS AAA EOS BOS BBBBBBBBBB EOS
        然后标签就是AAA EOS BOS BBBBBBBBBB EOS PAD
        正好是一个对应关系
    Args:
        batch (dict): dataset处理的内容

    Returns:
        dict: 返回训练的句子和标签
    """
    del_save = []
    for i in range(len(batch)):
        if len(batch[i]["content"]) > 150:
            del_save.append(i)
    if len(del_save) > 0:
        for i in range(len(del_save) - 1, -1, -1):
            del batch[del_save[i]]

    max_length_content = 0
    content_ = []
    for item in batch:
        content_.append(item["title"] + item["content"])

        if len(content_[len(content_) - 1]) > max_length_content:
            max_length_content = len(content_[len(content_) - 1])

    for index in range(len(content_)):
        content_[index] += ([2] * (max_length_content - len(content_[index])))

    label = content_.copy()
    content_ = torch.tensor(content_, dtype=torch.int64)
    for i in range(len(label)):
        del label[i][0]
        label[i].append(2)
    label = torch.tensor(label, dtype=torch.int64)
    return {
        "content": content_,
        "label": label
    }

def GetLoader(file=None, key_path=None):
    """获取Loader

    Args:
        file (str, optional): 需要loader的内容的路径. Defaults to None.
        key_path (str, optional): 生成的字典. Defaults to None.

    Returns:
        _type_: loader
    """
    if not file:
        file = os.path.join(os.getcwd(), r'chinese-poetry', 'quan_tang_shi', 'json')
    if not key_path:
        key_path = os.path.join(os.getcwd(), r'Dataset/key_.json')
    return DataLoader(
        dataset=TSdataset(file, key_path),
        batch_size=16,
        shuffle=True,

        collate_fn=collate_fn
    )

def GetLoaderRnn(file=None, key_path=None):
    if not file:
        file = os.path.join(os.getcwd(), r'chinese-poetry', 'quan_tang_shi', 'json')
    if not key_path:
        key_path = os.path.join(os.getcwd(), r'Dataset/key_.json')
    return DataLoader(
        dataset=TSdataset(file, key_path),
        batch_size=128,
        shuffle=True,
        collate_fn=collate_fn_rnn
    )


if __name__ == "__main__":
    loader = GetLoaderRnn()

    for i, data in enumerate(loader):
        print(data)