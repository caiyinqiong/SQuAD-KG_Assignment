import json
import random
import torch
import tqdm
from allennlp.modules.elmo import batch_to_ids
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper

class BatchGen(object):
    def __init__(self, data_path, batch_size, is_train):
        self.data_path = data_path
        self.batch_size = batch_size
        self.is_train = is_train
        self.data = self.load(self.data_path, is_train)  ## data是一维列表

        if is_train:
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            data = [self.data[i] for i in indices]
        data = [self.data[i:i+self.batch_size] for i in range(0, len(self.data), self.batch_size)]   ## data是二维列表，每一行是一个batch的数据
        self.data = data
        self.offset = 0

    ## load data 到一维数组
    def load(self, data_path, is_train):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = []
            for line in f:
                sample = json.loads(line)
                if is_train and (sample['start'] is None or sample['end'] is None or len(sample['doc_tok'])>1000):
                    continue
                data.append(sample)
        return data

    ## 打乱batch的顺序
    def reset(self):
        if self.is_train:
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            self.data = [self.data[i] for i in indices]
        self.offset = 0

    def __len__(self):
        return len(self.data)

    def __random_select__(self, arr):
        return [1 if random.uniform(0, 1) < 0.05 else e for e in arr]   ## dropout，1 是UNKUNK_ID

    def __iter__(self):
        while self.offset < len(self):
            batch = self.data[self.offset]  
            batch_size = len(batch)
            batch_dict = {}

            # doc
            doc_len = max(len(x['doc_tok']) for x in batch)  ## batch 内最长的文本长度    
            doc_id = torch.LongTensor(batch_size, doc_len).fill_(0)
            doc_tag = torch.LongTensor(batch_size, doc_len).fill_(0)
            doc_ent = torch.LongTensor(batch_size, doc_len).fill_(0)
            feature_len = len(eval(batch[0]['doc_fea'])[0]) if len(batch[0].get('doc_fea', [])) > 0 else 0  ## 4 (词频比例、精确匹配、小写匹配、lemma匹配)
            doc_feature = torch.Tensor(batch_size, doc_len, feature_len).fill_(0)
            doc_cid = torch.LongTensor(batch_size, doc_len, ELMoCharacterMapper.max_word_length).fill_(0)   # elmo

            # query
            query_len = max(len(x['query_tok']) for x in batch)    ## batch 内最长的query长度 
            query_id = torch.LongTensor(batch_size, query_len).fill_(0)
            query_cid = torch.LongTensor(batch_size, query_len, ELMoCharacterMapper.max_word_length).fill_(0)   # elmo 

            for i, sample in enumerate(batch):
                ## doc （的特征包括：token id、词性、实体、词频比例、精确匹配、小写匹配、lemma匹配）
                doc_tok = sample['doc_tok']
                doc_select_len = min(len(doc_tok), doc_len)               
                # if self.is_train:
                #     doc_tok = self.__random_select__(doc_tok)         ## mask                         
                doc_id[i, :doc_select_len] = torch.LongTensor(doc_tok[:doc_select_len])
                doc_tag[i, :doc_select_len] = torch.LongTensor(sample['doc_pos'][:doc_select_len])
                doc_ent[i, :doc_select_len] = torch.LongTensor(sample['doc_ner'][:doc_select_len])
                # 词频比例、精确匹配、小写匹配、lemma匹配
                for j, feature in enumerate(eval(sample['doc_fea'])):   
                    if j >= doc_select_len:  
                        break
                    doc_feature[i,j,:] = torch.Tensor(feature) 
                # elmo      
                doc_ctok = sample['doc_ctok']
                for j, w in enumerate(batch_to_ids(doc_ctok)[0].tolist()):    
                    if j >= doc_select_len:
                        break
                    doc_cid[i, j, :len(w)] = torch.LongTensor(w)

                ## query(的特征包括：token id、elmo)
                query_tok = sample['query_tok']
                query_select_len = min(len(query_tok), query_len)
                # if self.is_train: 
                #     query_tok = self.__random_select__(query_tok) 
                query_id[i, :query_select_len] = torch.LongTensor(query_tok[:query_select_len])
                # elmo
                query_ctok = sample['query_ctok']
                for j, w in enumerate(batch_to_ids(query_ctok)[0].tolist()):
                    if j >= query_select_len:
                        break
                    query_cid[i, j, :len(w)] = torch.LongTensor(w)

            batch_dict['uids'] = [sample['uid'] for sample in batch]           
            batch_dict['doc_tok'] = doc_id
            batch_dict['doc_mask'] = torch.eq(doc_id, 1)   
            batch_dict['doc_pos'] = doc_tag
            batch_dict['doc_ner'] = doc_ent
            batch_dict['doc_fea'] = doc_feature
            batch_dict['doc_ctok'] = doc_cid       # elmo
            batch_dict['query_tok'] = query_id
            batch_dict['query_mask'] = torch.eq(query_id, 1)
            batch_dict['query_ctok'] = query_cid   # elmo
            if self.is_train:
                batch_dict['start'] = torch.LongTensor([sample['start'] for sample in batch])
                batch_dict['end'] = torch.LongTensor([sample['end'] for sample in batch])
                batch_dict['label'] = torch.FloatTensor([sample['label'] for sample in batch])
            else:
                batch_dict['text'] = [sample['context'] for sample in batch]
                batch_dict['span'] = [sample['span'] for sample in batch]

            self.offset += 1
            yield batch_dict



class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, value, n):
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count



def predict_squda(model, dev_data):
    dev_data.reset()
    answer_predictions = {}
    for batch in tqdm.tqdm(dev_data, total=len(dev_data)): 
        uids = batch['uids']
        answer_preds, _ = model.predict(batch)
        for uid, answer in zip(uids, answer_preds):
            answer_predictions[uid] = answer
            
    return answer_predictions
