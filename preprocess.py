import argparse
import pickle
import os
import json
import tqdm
import unicodedata
import numpy as np
import spacy
from collections import Counter

from my_utils.tokenizer import Vocabulary
from my_utils.utils import reform_text
from my_utils.prepro_utils import feature_func


def load_data(data_path, is_train):
    rows = []
    with open(data_path, encoding='utf-8') as f:
        data = json.load(f)['data']    ## [:100] ################################ 测试代码用
    
    for article in tqdm.tqdm(data, total=len(data)):
        for paragraph in article['paragraphs']:
            context = '{} {}'.format(paragraph['context'], 'EOSEOS')
            for qa in paragraph['qas']:
                uid = qa['id'] 
                question = qa['question']
                answers = qa.get('answers', [])
                is_impossible = qa.get('is_impossible', False)
                label = 1 if is_impossible else 0     ## label 表示问题是否不可回答

                if is_train:  ## 对训练集数据
                    if label == 0 and len(answers) == 0:   ## 如果问题可回答，但答案长度为0
                        continue
                    if len(answers) > 0:
                        answer = answers[0]['text']
                        answer_start = answers[0]['answer_start']
                        answer_end = answer_start + len(answer)
                        sample = {'uid': uid,
                                  'context': context,
                                  'question': question,
                                  'answer': answer,
                                  'answer_start': answer_start,
                                  'answer_end': answer_end,
                                  'label': label}
                    else:   # 如果问题不可回答，且答案为空
                        answer = 'EOSEOS'
                        answer_start = len(context) - len('EOSEOS')
                        answer_end = len(context)
                        sample = {'uid': uid,
                                  'context': context,
                                  'question': question,
                                  'answer': answer,
                                  'answer_start': answer_start,
                                  'answer_end': answer_end,
                                  'label': label}
                else:   ## 对开发集数据
                    sample = {'uid': uid,
                              'context': context,
                              'question': question,
                              'answer': answers,
                              'answer_start': -1,
                              'answer_end': -1}
                
                rows.append(sample)
    return rows

def load_glove_vocab(glove_data_path):
    vocab = set()
    with open(glove_data_path, encoding='utf-8') as f:
        for line in f:
            elems = line.split()           
            token = unicodedata.normalize('NFD', ' '.join(elems[0:-300]))   # unicodedata.normalize（）将文本标准化, 第一个参数表示字符串标准化的方式
            vocab.add(token)
    return vocab

def build_squda_vocab(squda_data, glove_vocab):
    nlp = spacy.load('en_core_web_sm', disable=['vectors', 'textcat', 'parser'])

    docs = [reform_text(sample['context']) for sample in squda_data]
    doc_tokened = [doc for doc in nlp.pipe(docs, batch_size=1000, n_threads=24)]    ### 
    questions = [reform_text(sample['question']) for sample in squda_data]
    question_tokened = [question for question in nlp.pipe(questions, batch_size=10000, n_threads=24)] 
    
    question_counter = Counter()
    doc_counter = Counter()

    for tokened in tqdm.tqdm(doc_tokened, total=len(doc_tokened)):
        doc_counter.update([unicodedata.normalize('NFD', w.text) for w in tokened if len(unicodedata.normalize('NFD', w.text)) > 0])
    for tokened in tqdm.tqdm(question_tokened, total=len(question_tokened)):
        question_counter.update([unicodedata.normalize('NFD', w.text) for w in tokened if len(unicodedata.normalize('NFD', w.text)) > 0]) 
    
    counter = question_counter + doc_counter   #相同单词的会合并
    # vocab存储可以匹配上的单词，question中的词汇在前，doc的在后，按词频排
    vocab = sorted([w for w in question_counter if w in glove_vocab], key=question_counter.get, reverse=True)  ##sort是内置，sorted返回一个副本；reverse=True表示降序排列
    vocab += sorted([w for w in doc_counter.keys() - question_counter.keys() if w in glove_vocab], key=counter.get, reverse=True)
    print('数据集中总词汇数：' + str(len(counter)))
    print('数据集中可以在glove中匹配的词汇数：' + str(len(vocab)))

    total = sum(counter.values()) 
    matched = sum(counter[w] for w in vocab)
    print('数据集中总单词数：' + str(total))
    print('数据集中可以在glove中匹配的单词数：' + str(matched))
    print('数据集中OOV的单词数：' + str(total - matched))

    vocab = Vocabulary.build(vocab)
    return vocab

# 根据glove文件位置 和 squda的词汇表 处理得到二维数组embedding
def build_squda_embedding(glove_data_path, squda_vocab):
    emb = np.zeros((len(squda_vocab), 300))
    with open(glove_data_path, encoding='utf-8') as f:
        for line in f:
            elems = line.split()
            token = unicodedata.normalize('NFD', ' '.join(elems[0:-300]))
            if token in squda_vocab:
                emb[squda_vocab[token]] = [float(v) for v in elems[-300:]]
    return emb

def build_data(data, vocab, vocab_tag, vocab_ner, file_out, is_train):
    nlp = spacy.load('en_core_web_sm', disable=['vectors', 'textcat', 'parser'])

    docs = [reform_text(sample['context']) for sample in data]
    docs_tokened = [doc for doc in nlp.pipe(docs, batch_size=1000, n_threads=24)]
    questions = [reform_text(sample['question']) for sample in data]
    questions_tokened = [question for question in nlp.pipe(questions, batch_size=1000, n_threads=24)]

    with open(file_out, 'w', encoding='utf-8') as f:
        for idx, sample in enumerate(data):
            if idx % 1000 == 0:
                print('正在处理第 ' + str(idx) + ' 个sample')
            feature_dict = feature_func(sample, questions_tokened[idx], docs_tokened[idx], vocab, vocab_tag, vocab_ner, is_train)
            if feature_dict is not None:
                f.write('{}\n'.format(json.dumps(feature_dict)))

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', default='F:\\0recentwork\\SQuDA\\data\\train-v2.0.json')
    parser.add_argument('--dev_data_path', default='F:\\0recentwork\\SQuDA\\data\\dev-v2.0.json')
    parser.add_argument('--glove_data_path', default='F:\\0recentwork\\DIIN-cyq\\datasets\\glove\\glove.840B.300d.txt')
    parser.add_argument('--vocab_tag_data_path', default='F:\\0recentwork\\SQuDA\\data\\vocab_tag.pick')
    parser.add_argument('--vocab_ner_data_path', default='F:\\0recentwork\\SQuDA\\data\\vocab_ner.pick')
    parser.add_argument('--out_dir', default='F:\\0recentwork\\SQuDA\\output')   ## 输出文件目录
    args = parser.parse_args()
    return args

def main():
    args = set_args()

    print('start load data')
    train_data = load_data(args.train_data_path, is_train=True)
    print('训练集的问答对总数：' + str(len(train_data)))
    dev_data = load_data(args.dev_data_path, is_train=False)
    print('开发集的问答对总数：' + str(len(dev_data)))
    ## load 其他特征词典
    vocab_tag_file = open(args.vocab_tag_data_path, 'rb')
    vocab_tag = pickle.load(vocab_tag_file)
    vocab_tag_file.close()
    vocab_ner_file = open(args.vocab_ner_data_path, 'rb')
    vocab_ner = pickle.load(vocab_ner_file)
    vocab_ner_file.close()
    print('load_data end!')

    # load glove
    print('start load glove vocabulary')
    glove_vocab = load_glove_vocab(args.glove_data_path)    
    print('load glove vocabulary end')

    # 处理squda的词汇表和相应的embedding
    print('start build squda vocabulary and embedding')
    squda_vocab = build_squda_vocab(train_data + dev_data, glove_vocab)  ## 得到一个Vocabulary
    squda_embedding = build_squda_embedding(args.glove_data_path, squda_vocab)
    print('build squda vocabulary and embedding end')

    # 把所有的词典（squda_vocab，vacab_tag，vocab_ner）和 squda_embedding 写出
    print('开始写squda_v2.pick文件')
    data = {'vocab': squda_vocab, 
            'vocab_tag': vocab_tag,
            'vocab_ner': vocab_ner,
            'embedding': squda_embedding}
    with open(os.path.join(args.out_dir, 'squda_v2.pick'), 'wb') as f:
        pickle.dump(data, f)  # dump 是依次将这些写入文件
    print('写squda_v2.pick文件结束')

    # 处理 train 数据
    print('start build train data')
    train_out_file_path = os.path.join(args.out_dir, 'train_preprocess.json')
    build_data(train_data, squda_vocab, vocab_tag, vocab_ner, train_out_file_path, is_train=True)

    # 处理 dev 数据
    print('start build dev data')
    dev_out_file_path = os.path.join(args.out_dir, 'dev_preprocess.json')
    build_data(dev_data, squda_vocab, vocab_tag, vocab_ner, dev_out_file_path, is_train=False)


if __name__ == "__main__":
    main()
    print('done')