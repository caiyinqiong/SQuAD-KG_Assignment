import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable 
import torch.nn.functional as F
import numpy as np
import math

from my_utils.train_utils import AverageMeter
from .sanQA_network import SANQA_Network


class SANQA_Model(object):
    def __init__(self, opt, embedding):
        self.opt = opt
        self.updates = 0   ## 记录一个epoch训练了多少batch
        self.eval_embed_transfer = True  
        self.train_loss = AverageMeter()
        self.network = SANQA_Network(opt, embedding)  
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        self.optimizer = optim.Adamax(parameters, opt['learning_rate'])
        freezed_vec_size = (opt['vocab_size'] -opt['embed_tune_partial']) * 300  ## 冻结的参数量
        self.total_param = sum([p.nelement() for p in parameters]) - freezed_vec_size    ## p.nelement()返回张量中的元素个数

    def setup_eval_embed(self, eval_embed):  ## 
        self.network.lexicon_encoder.eval_embed = nn.Embedding(eval_embed.size(0), eval_embed.size(1), padding_idx=0)
        self.network.lexicon_encoder.eval_embed.weight.data = eval_embed
        for p in self.network.lexicon_encoder.eval_embed.parameters():
            p.requires_grad = False
        self.eval_embed_transfer = True 
        self.network.lexicon_encoder.ContextualEmbed.setup_eval_embed(eval_embed)
    
    def update_eval_embed(self):  
        if self.opt['embed_tune_partial'] > 0:
            offset = self.opt['embed_tune_partial']
            self.network.lexicon_encoder.eval_embed.weight.data[0:offset] = self.network.lexicon_encoder.embedding.weight.data[0:offset]

    ## 一个batch训练模型
    def update(self, batch):
        self.network.train()  
        y = Variable(batch['start']), Variable(batch['end'])
        label = Variable(batch['label'], requires_grad=False)

        start, end, pred = self.network(batch)   
        loss = F.nll_loss(start, y[0]) + F.nll_loss(end, y[1])  ## span   
        loss += F.binary_cross_entropy(pred, torch.unsqueeze(label, 1))    ## label

        self.train_loss.update(loss.item(), len(start))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 5)  ## 梯度裁剪，设置最大范数为5
        self.optimizer.step()
        self.updates += 1
        self.reset_embeddings()  
        self.eval_embed_transfer = True  
        
    ## 对于冻结的 embedding 参数要重置
    def reset_embeddings(self):
        if self.opt['embed_tune_partial'] > 0:
            offset = self.opt['embed_tune_partial']
            if offset < self.network.lexicon_encoder.embedding.weight.data.size(0):
                self.network.lexicon_encoder.embedding.weight.data[offset:] = self.network.lexicon_encoder.fixed_embedding  

    def predict(self, batch):
        self.network.eval()  
        if self.eval_embed_transfer:
            self.update_eval_embed()  ## 用训练时候更新过的embedding来更新eval_embd
            self.eval_embed_transfer = False
        
        # 预测输出
        start, end, label = self.network(batch)
        start = F.softmax(start, 1)
        end = F.softmax(end, 1)
        start = start.data.cpu()  
        end = end.data.cpu()
        label = label.data.cpu()
        
        text = batch['text']
        spans = batch['span']
        span_predictions = []
        label_predictions = []
 
        for i in range(start.size(0)):
            # label
            label_score = float(label[i])   ## 有没有答案的概率
            label_predictions.append(label_score)
            # answer span
            if label_score > 0.5:
                answer = ''
            else:
                max_score = 0
                max_score_sidx = 0
                max_score_eidx = 0
                for s_idx in range(0, len(spans[i]) - 1):
                    for e_idx in range(s_idx, min(s_idx+self.opt['max_len'], len(spans[i]) - 1)):
                        score = start[i][s_idx] * end[i][e_idx]
                        if score > max_score:
                            max_score = score
                            max_score_sidx = s_idx
                            max_score_eidx = e_idx
                s_offset, e_offset = spans[i][max_score_sidx][0], spans[i][max_score_eidx][1]
                answer = text[i][s_offset:e_offset] 
            span_predictions.append(answer) 
        return (span_predictions, label_predictions)

    def save(self, filename):
        network_state = dict([(k, v) for k, v in self.network.state_dict().items() 
                            if k.find('ContextualEmbed') == -1 and k.find('fixed_embedding') == -1 and k.find('eval_embed.weight') == -1])              
        params = {'state_dict': {'network': network_state}, 'config': self.opt}
        torch.save(params, filename)
