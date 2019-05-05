import pandas as pd
import re
from collections import defaultdict
import json
import numpy as np
import time

# STOP_CHAR = '[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）0-9a-zA-Z]+'
# SPLIT_CHAR = re.compile("([\u4E00-\u9FD5a-zA-Z0-9-+#&\._/\u03bc\u3001\(\)\uff08\uff09\~\'\u2019]+)", re.U)
STOP_CHAR = '[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）0-9a-zA-Z]+'
SPLIT_CHAR = re.compile("([\u4E00-\u9FD5a-zA-Z0-9-+#&\._/\u03bc\u3001\(\)\uff08\uff09\~\'\u2019]+)", re.U)

def read(corpus_file):
    
    texts = None
    if corpus_file.endswith(".txt"):
        with open(corpus_file, encoding='utf-8') as f:
            texts = f.readlines()
    elif corpus_file.endswith(".csv"):
        df = pd.read_csv(corpus_file)
        texts = list(df[text_col_name])
    elif corpus_file.endswith(".xlsx"):
        df = pd.read_csv(corpus_file)
        texts = list(df[text_col_name])
    else:
        print("illegal corpus format:" + corpus_file)
    sentences = []
    for text in texts:
        segs = SPLIT_CHAR.findall(text.strip())
        for seg in segs:
            sentences.append(re.sub(STOP_CHAR, "", seg))
    return sentences


def gen_count_info(sentences, ngram=4):
    cnt_dict = {}
    total_cnt = 0
    for sent in sentences:
        N = len(sent)
        for i in range(N):
            for n in range(1, ngram + 1):
                end = i + n
                if end > N:
                    continue
                total_cnt += 1
                token = sent[i:end]
                if token not in cnt_dict:
                    cnt_dict[token] = defaultdict(int)
                    cnt_dict[token]["cnt"] = 0
                    cnt_dict[token]["left"] = defaultdict(int)
                    cnt_dict[token]["right"] = defaultdict(int)
                cnt_dict[token]["cnt"] += 1
                if i == 0:
                   cnt_dict[token]["left"]['<S>'] += 1
                else:
                   cnt_dict[token]["left"][sent[i - 1]] += 1
                   
                if end == N:
                    cnt_dict[token]["right"]['<E>'] += 1
                else:
                    cnt_dict[token]["right"][sent[end]] += 1
    return cnt_dict, total_cnt
    
def filter_by_freq(cnt_dict, min_freq=10):
    filtered_dict = {}
    for k, v in cnt_dict.items():
        if v['cnt'] >= min_freq:
            filtered_dict[k] = v
    return filtered_dict

def filter_by_blacklist(cnt_dict, black_list_file="../data/black_list.txt"):
    black_list = None
    filtered_dict = {}
    with open(black_list_file, encoding='utf-8') as f:
        texts = f.readlines()
        black_list = set(texts)
    for k, v in cnt_dict.items():
        if k not in black_list:
            filtered_dict[k] = v
    return filtered_dict

def cal_entropy(cnt_item):
    cnts = []
    for k, v in cnt_item.items():
        cnts.append(v)
    cnts = np.array(cnts)
    p = cnts / cnts.sum()
    entropy = (-p * np.log2(p)).sum()
    return entropy

def cal_margin_entropy(cnt_dict):
    for k, v in cnt_dict.items():
        cnt_dict[k]['left'] = cal_entropy(v['left'])
        cnt_dict[k]['right'] = cal_entropy(v['right'])

def cal_ngram_pmi(cnt_dict, total_cnt):
    for token, v in cnt_dict.items():
        #only consider 2gram or larger
        if len(token) > 1:
            pmi_list = [total_cnt * v['cnt'] / (cnt_dict[token[:i + 1]]['cnt'] * cnt_dict[token[i + 1:]]['cnt'])  for i in range(len(token) - 1)]
            score = min(pmi_list)

            cnt_dict[token]['pmi'] = score
    

def filter_by_pmi(cnt_dict, min_pmi):
    filtered_dict = {}
    for k, v in cnt_dict.items():
        if v['pmi'] > min_pmi[len(k)]:
            filtered_dict[k] = v
    return filtered_dict

def filter_by_entropy(cnt_dict, min_entropy=1.0):
    filtered_dict = {}
    for k, v in cnt_dict.items():
        if v['left'] > min_entropy and v['right'] > min_entropy:
            filtered_dict[k] = v
    return filtered_dict
        
def save(cnt_dict, output="../result/cnt_info.txt"):
    file = open(output, 'w', encoding="utf-8")
    for k, v in cnt_dict.items():
        file.write("{}\t{}\n".format(k, v))
    return

def test():
    min_pmi={1:0, 2:5, 3:25, 4:125}
    
    print("Starting to read!")
    sents = read("../data/corpus_finance.txt")
    cnt_dict, total_cnt = gen_count_info(sents)
    print("calculate pmi...")
    cal_ngram_pmi(cnt_dict, total_cnt)
    
    print("filtering by pmi...")
    cnt_dict = filter_by_pmi(cnt_dict, min_pmi)
    print("filtering by frequence...")
    cnt_dict = filter_by_freq(cnt_dict)
    print("filtering by black list...")
    cnt_dict = filter_by_blacklist(cnt_dict)
    
    
    file = open('../result/cnt_info.js', 'w')
    out_data = json.dumps(cnt_dict, ensure_ascii=False, indent=2)
    file.write(out_data)
    
    print("calculate margin entropy...")
    cal_margin_entropy(cnt_dict)
    
    print("filtering by entropy...")
    cnt_dict = filter_by_entropy(cnt_dict)
    
    print("find {} new tokens...".format(len(cnt_dict)))
    file = open('../result/entropy_info.js', 'w')
    out_data = json.dumps(cnt_dict, ensure_ascii=False, indent=2)
    file.write(out_data)
    
if __name__ == "__main__":
    start = time.clock()
    test()
    end = time.clock()
    print('Running time: %s Seconds'%(end - start))
    