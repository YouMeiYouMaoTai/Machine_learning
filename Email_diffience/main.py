import os
import re
import string
import numpy as np

# 过滤数字,把数字替换成 ‘’ 即：啥也没有
def replace_num(txt_str):
    txt_str = txt_str.replace(r'0', '')
    txt_str = txt_str.replace(r'1', '')
    txt_str = txt_str.replace(r'2', '')
    txt_str = txt_str.replace(r'3', '')
    txt_str = txt_str.replace(r'4', '')
    txt_str = txt_str.replace(r'5', '')
    txt_str = txt_str.replace(r'6', '')
    txt_str = txt_str.replace(r'7', '')
    txt_str = txt_str.replace(r'8', '')
    txt_str = txt_str.replace(r'9', '')
    return txt_str

# 对读取到的数据进行处理
def disposal(txt_str):
    translator = re.compile('[%s]' % re.escape(string.punctuation))
    # 全部小写
    txt_str = txt_str.lower()
    # 过滤掉所有符号
    txt_str = translator.sub(' ', txt_str)
    # 过滤掉全部数字
    txt_str = replace_num(txt_str)
    # 把全体的邮件文本 根据换行符把string划分成列表
    txt_str_list = txt_str.splitlines()
    # 把获取的全体单词句子列表转成字符串
    txt_str = ''.join(txt_str_list)
    return txt_str

def get_filtered_str(category):

    email_list = []
    # 把标点符号都汇聚起来
    translator = re.compile('[%s]' % re.escape(string.punctuation))
    for curDir, dirs, files in os.walk(f'D:\\Desktop\\Program\\Python\\Email_diffience\\Emails\\{category}'):
        for file in files:
            file_name = os.path.join(curDir, file)
            with open(file_name, 'r', encoding='utf-8') as f:
                txt_str = f.read()
                txt_str = disposal(txt_str)
            email_list.append(txt_str)
    return email_list

def get_dict_spam_dict_w(spam_email_list):
    all_email_words = []
    # 用set集合去重
    word_set = set()
    for email_str in spam_email_list:
        # 把每个邮件的文本 变成单词
        email_words = email_str.split(' ')
        # 把每个邮件去重后的列表 存入列表
        all_email_words.append(email_words)
        for word in email_words:
            if(word!=''):
                word_set.add(word)
    # 计算每个垃圾词出现的次数
    word_dict = {}
    for word in word_set:
        # 创建字典元素 并让它的值为1
        word_dict[word] = 0
        # 遍历每个邮件，看文本里面是否有该单词
        for email_words in all_email_words:
            for email_word in email_words:
                # 把从set中取出的word 和 每个email分词后的word对比看是否相等
                if(word==email_word):
                    word_dict[word] += 1
                    # 找到一个就行了
                    break
    return word_dict

def get_dict_ham_dict_w(spam_email_list,normal_email_list):
    all_ham_email_words = []
    # 用set集合去重 得到垃圾邮件的特征w
    word_set = set()
    #获取垃圾邮件特征
    for email_str in spam_email_list:
        # 把每个邮件的文本 变成单词
        email_words = email_str.split(' ')
        for word in email_words:
            if (word != ''):
                word_set.add(word)
    for ham_email_str in normal_email_list:
        # 把每个邮件的文本 变成单词
        ham_email_words = ham_email_str.split(' ')
        # print(f'ham_email_words={ham_email_words}')
        # 把每个邮件分割成单词的 的列表 存入列表
        all_ham_email_words.append(ham_email_words)

    # 计算每个垃圾词出现的次数
    word_dict = {}
    for word in word_set:
        # 创建字典元素 并让它的值为1
        word_dict[word] = 0

        # 遍历每个邮件，看文本里面是否有该单词
        for email_words in all_ham_email_words:
            # print(f'ham_email_words={email_words}')
            for email_word in email_words:
                # 把从set中取出的word 和 每个email分词后的word对比看是否相等
                # print(f'email_word={email_word}')
                if(word==email_word):
                    word_dict[word] += 1
                    # 找到一个就行了
                    break
    return word_dict

# 假设测试邮件是垃圾邮件，根据垃圾邮件中垃圾特征计算此时测试邮件是垃圾邮件的概率
def get_X_c1(spam_w_dict,file_name):

    translator = re.compile('[%s]' % re.escape(string.punctuation))
    with open(file_name, 'r', encoding='utf-8') as f:
        txt_str = f.read()
        txt_str = disposal(txt_str)
    # 把句子分成词
    email_words = txt_str.split(' ')

    # 去重
    x_set = set()
    for word in email_words:
        if word!='':
            x_set.add(word)
    spam_len = len(os.listdir(f'D:\\Desktop\\Program\\Python\\Email_diffience\\Emails\\spam'))
    # 判断测试邮件的词有哪些是垃圾邮件的特征
    spam_X_num = []
    for xi in x_set:
        for wi in spam_w_dict:
            if xi == wi:
                spam_X_num.append(spam_w_dict[wi])
    w_appear_sum_num = 1
    for num in spam_X_num:
        w_appear_sum_num += num
    # 求概率
    w_c1_p = w_appear_sum_num / (spam_len + 2)
    return w_c1_p

# 假设测试邮件是正常邮件，根据正常邮件中垃圾特征计算此时测试邮件是正常邮件的概率
def get_X_c2(normal_w_dict,file_name):

    # 过滤文本
    translator = re.compile('[%s]' % re.escape(string.punctuation))
    with open(file_name, 'r', encoding='utf-8') as f:
        txt_str = f.read()
        txt_str = disposal(txt_str)
    # 把句子分成词
    email_words = txt_str.split(' ')

    # 去重
    x_set = set()
    for word in email_words:
        if word!='':
            x_set.add(word)

    # 判断测试邮件的词有哪些是垃圾邮件的特征
    ham_X_num = []
    for xi in x_set:
        for wi in normal_w_dict:
            if xi == wi:
                ham_X_num.append(normal_w_dict[wi])

    # 先求分子 所有词出现的总和
    ham_len = len(os.listdir(f'D:\\Desktop\\Program\\Python\\Email_diffience\\Emails\\normal'))
    w_appear_sum_num = 1
    for num in ham_X_num:
        w_appear_sum_num += num
    # 求概率
    w_c2_p = w_appear_sum_num / (ham_len+2)
    return w_c2_p

def email_test(spam_w_dict,normal_w_dict):
    for curDir, dirs, files in os.walk(f'D:\\Desktop\\Program\\Python\\Email_diffience\\Emails\\test'):
        for file in files:
            file_name = os.path.join(curDir, file)
            print('===============================')
            print(f'测试邮件: text\\{file}')
            # 获取条件概率 p(X|c1)
            p_X_c1 = get_X_c1(spam_w_dict,file_name)
            # 获取条件概率 p(X|c2)
            p_X_c2 = get_X_c2(normal_w_dict,file_name)
            # #注意：Log之后全部变为负数
            # print('{:.4f}'.format(pi_10))
            A = np.log(p_X_c1) + np.log(1 / 2)
            B = np.log(p_X_c2) + np.log(1 / 2)
            # 除法会出现问题，-1 / 负分母  结果 < -2/同一个分母
            print(f'垃圾邮件的概率为：p1={round(A, 5)}\n正常邮件的概率为：p2={round(B, 5)}')

            # 因为分母一致，所以只比较 分子即可
            if A > B:
                print('p1>p2，所以是垃圾邮件.')
            if A <= B:
                print('p1<p2，所以是正常邮件.')

if __name__=='__main__':
    spam_email_list = get_filtered_str('spam')
    normal_email_list = get_filtered_str('normal')
    spam_w_dict = get_dict_spam_dict_w(spam_email_list)
    normal_w_dict = get_dict_ham_dict_w(spam_email_list,normal_email_list)
    print(f'\n从垃圾邮件中提取的特征及每个特征出现的邮件数：')
    print(f'{spam_w_dict}')
    print(f'\n普通邮件中垃圾邮件特征出现的邮件数为：')
    print(f'{normal_w_dict}')
    email_test(spam_w_dict, normal_w_dict)