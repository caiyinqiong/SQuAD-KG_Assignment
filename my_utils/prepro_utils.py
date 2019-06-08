from collections import Counter
import numpy as np
import re


## feature包括 词频占比、原始匹配、小写形式匹配、词干匹配 四个信息
def match_func(question_tokened, doc_tokened):
    question_word = {w.text for w in question_tokened}
    question_lower = {w.text.lower() for w in question_tokened}
    question_lemma = {w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in question_tokened}  ## 代词保留词本身

    counter = Counter(w.text.lower() for w in doc_tokened)
    total = sum(counter.values())
    freq = [counter[w.text.lower()]/total for w in doc_tokened]

    match_origin = [1 if w in question_word else 0 for w in doc_tokened]
    match_lower = [1 if w.text.lower() in question_lower else 0 for w in doc_tokened]
    match_lemma = [1 if (w.lemma_ if w.lemma_!='-PRON-' else w.text.lower()) in question_lemma else 0 for w in doc_tokened]
    
    features = np.array([freq, match_origin, match_lower, match_lemma], dtype=np.float).T.tolist()
    return features

## 返回三元组（t_start表示答案开始的token下标， t_end表示答案结束的token下标，t_span表示doc_toks中每个token的起止字符下标）
def build_span(context, answer, doc_toks, answer_start, answer_end, is_train):
    p_str = 0
    p_token = 0
    t_start, t_end, t_span = -1, -1, []
    while p_str < len(context):
        if re.match(r'\s', context[p_str]):
            p_str += 1
            continue
        token = doc_toks[p_token]
        token_len = len(token)
        t_span.append((p_str, p_str + token_len))
        if is_train:
            if (p_str <= answer_start and p_str + token_len > answer_start):
                t_start = p_token
            if (p_str < answer_end and p_str + token_len >= answer_end):
                t_end = p_token
        p_str += token_len
        p_token += 1
    if is_train and (t_start == -1 or t_end == -1):
        return (-1, -1, [])
    else:
        return (t_start, t_end, t_span)


def feature_func(sample, question_tokened, doc_tokened, vocab, vocab_tag, vocab_ner, is_train):
    feature_dict = {}
    feature_dict['uid'] = sample['uid']  ## 问题id
    if is_train:
        feature_dict['label'] = sample['label']
    
    query_toks = [token.text for token in question_tokened if len(token.text)>0]
    feature_dict['query_ctok'] = query_toks
    feature_dict['query_tok'] = [vocab[w.text] for w in question_tokened if len(w.text)>0]        # 得到token_list对应的词id列表
    feature_dict['query_pos'] = [vocab_tag[w.tag_] for w in question_tokened if len(w.text)>0]    # 得到token_list对应的词性信息列表
    feature_dict['query_ner'] = [vocab_ner['{}_{}'.format(w.ent_type_, w.ent_iob_)] for w in question_tokened if len(w.text)>0]  # 实体信息

    doc_toks = [token.text for token in doc_tokened if len(token.text)>0]
    feature_dict['doc_ctok'] = doc_toks
    feature_dict['doc_tok'] = [vocab[w.text] for w in doc_tokened if len(w.text)>0] 
    feature_dict['doc_pos'] = [vocab_tag[w.tag_] for w in doc_tokened if len(w.text)>0] 
    feature_dict['doc_ner'] = [vocab_ner['{}_{}'.format(w.ent_type_, w.ent_iob_)] for w in doc_tokened if len(w.text)>0] 

    feature_dict['doc_fea'] = '{}'.format(match_func(question_tokened, doc_tokened))  # doc的匹配信息
    feature_dict['query_fea'] = '{}'.format(match_func(doc_tokened, question_tokened))  # query的匹配信息

    start, end, span = build_span(sample['context'], sample['answer'], doc_toks, sample['answer_start'], sample['answer_end'], is_train)
    feature_dict['start'] = start
    feature_dict['end'] = end
    if is_train and (start == -1 or end == -1):
        return None
    if not is_train:
        feature_dict['context'] = sample['context']
        feature_dict['span'] = span

    return feature_dict