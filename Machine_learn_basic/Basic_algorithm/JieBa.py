import jieba

content = "现在，机器学习和深度学习带动人工智能飞速的发展，并在图像处理、语音识别领域取得巨大成功。"

#精确分词：精确模式试图将句子最精确地切开，精确分词也是默认的分词方式
segs_1 = jieba.cut(content, cut_all=False)
print("/".join(segs_1))
print("-----------------------")

#全模式分词:把句子中所有的可能是词语的都扫描出来，速度非常快，但不能解决歧义。
segs_2 = jieba.cut(content, cut_all=True)
print("/".join(segs_2))
print("-----------------------")

#搜索引擎模式：在精确模式的基础上，对长词再次进行划分，提高召回率Recall，适用于搜索引擎分词。
segs_3 = jieba.cut_for_search(content)
print("/".join(segs_3))
print("-----------------------")

#用lcut生成list
segs_4 = jieba.lcut(content)
print(segs_4)
print("-----------------------")

#获取分词结果中词列表的 top N
from collections import Counter

top5 = Counter(segs_4).most_common(5)
print(top5)
print("-----------------------")

#自定义添加词和字典
txt = "字节跳动是中国一家新兴的互联网公司。"
segs_5 = jieba.lcut(txt)
print(segs_5)
print("-----------------------")

jieba.add_word("字节跳动")
segs_6 = jieba.lcut(txt)
print(segs_6)
print("-----------------------")

txt1 = "火山小视频是字节跳动公司开发的一款应用软件。"
jieba.load_userdict("user_dict.txt")  # user_dict.txt是自己创建的一个自定义的新词词典
segs_7 = jieba.lcut(txt1)
print(segs_7)
print("-----------------------")