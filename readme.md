# 基于神经网络的唐诗生成

数据源自[chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)

一共写了两个吧，一个就用LSTM写的，一个用transformer写的。
其实说实话效果还是可以的。

> 比如LSTM生成的：

请输入题目:於塞北春日思歸

题目

    于塞北春日思归

作诗
    北风吹雨雪，南陌几时春。草色连山色，风声似柳楼。山川多旧路，江海共长流。更忆东山下，春风满树愁。
> transformer生成的

请输入题目:於塞北春日思歸

题目

    於塞北春日思歸

作诗
    去年今日事，今日望鄉春。獨向春風起，歸思故園人。

其实看质量还是可以的？主要是生成的诗句感觉读着还可以但是意思就是差很多，或者有带你不连贯那种吧。我感觉还是数据量比较小，如果未来我可以即使只用一些控制的语句，比如风格之类的或许会跟好点，还有就是我想加入一些随机的向量来通过一句题目生成更多的诗句，就是增加他的多样性。

- [x] LSTM
- [x] TRANSFORMER
- [ ] 加入GAN
- [ ] 提高生成质量

