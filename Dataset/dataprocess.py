import json
import glob
import os
import zhconv


# 获取所有的文件
def readFIels(path):
    return glob.glob(os.path.join(path,"*"))

# 处理一个json文件中所有的唐诗
def processOneFile(file):
    '''
    return:[{'title':'','auth':'','content':[]},...]
    '''
    contents = []
    with open(file,"r",encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            title = item['title']
            paragraphs = item['paragraphs']
            author = item['author']
            contents.append({
                "auth":author,
                "content":paragraphs,
                "title":title
            })
        
    return contents




if __name__ == "__main__":
    ps = readFIels(os.getcwd() + os.sep + 'chinese-poetry/quan_tang_shi/json')

    # 统计字数，然后生成列表
    code_set = set()
    for ipath in ps:
        with open(ipath,"r") as f:
            data = json.load(f)
            for item in data:
                title = item['title']
                for code in title:
                    code_set.add(code)
                paragraphs = item['paragraphs']
                for line in paragraphs:
                    for code in line.strip():
                        code_set.add(code)
                author = item['author']
                for code in author:
                    code_set.add(code)
    
    # 这里没有设置UNK 主要是因为懒，而且没有分测试集，主要是因为分了好像也没啥用，所以训练的时候就不会出现UNK
    code_index = {"BOS":0,"EOS":1,"PAD":2}
    index_code = {0:"BOS",1:"EOS",2:"PAD"}
    codes = list(code_set)
    for i in range(3,len(codes) + 3):
        index_code[i] = codes[i - 3]
        code_index[codes[i - 3]] = i
    key_json = [index_code,code_index,len(codes) + 3]
    with open("key_.json","w+") as fcode:
        json.dump(key_json,fcode,ensure_ascii=False)