import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import random


class ContextualEmbed(nn.Module):
    def __init__(self, cove_path, vocab_size, embedding):
        super(ContextualEmbed, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding.size(-1), padding_idx=0)
        self.embedding.weight.data = embedding

        state_dict = torch.load(cove_path)
        self.rnn1 = nn.LSTM(300, 300, num_layers=1, bidirectional=True)
        self.rnn2 = nn.LSTM(600, 300, num_layers=1, bidirectional=True)
        state_dict1 = dict([(name, param.data) if isinstance(param, Parameter) else (name, param)
                            for name, param in state_dict.items() if '0' in name])
        state_dict2 = dict([(name.replace('1', '0'), param.data) if isinstance(param, Parameter) else (name.replace('1','0'), param)
                            for name, param in state_dict.items() if '1' in name])
        self.rnn1.load_state_dict(state_dict1)
        self.rnn2.load_state_dict(state_dict2)
        for p in self.parameters():   
            p.requires_grad = False
        self.output_size = 600

    def setup_eval_embed(self, eval_embed):  # model创建之后调用的
        self.eval_embed = nn.Embedding(eval_embed.size(0), eval_embed.size(1), padding_idx=0)
        self.eval_embed.weight.data = eval_embed
        for p in self.eval_embed.parameters():
            p.requires_grad = False

    def forward(self, x_idx):  ##输入是（doc_tok）或(query_tok) 
        emb = self.embedding if self.training else self.eval_embed
        x_hiddens = emb(x_idx)
        lengths = torch.Tensor([x_idx.size(1)]*x_idx.size(0)).long()
        max_len = x_idx.size(1)     ## token 数
        lens, indices = torch.sort(lengths, 0, True)
        packed_inputs = pack(x_hiddens[indices], lens.tolist(), batch_first=True)

        output1, _ = self.rnn1(packed_inputs)  
        output2, _ = self.rnn2(output1)
        output2 = unpack(output2, batch_first=True, total_length=max_len)[0]
        _, _indices = torch.sort(indices, 0)
        output2 = output2[_indices]
        return output2


class PositionwiseNN(nn.Module):
    def __init__(self, i_dim, h_dim):
        super(PositionwiseNN, self).__init__()
        self.proj_1 = nn.Linear(i_dim, h_dim)
        self.proj_2 = nn.Linear(h_dim, h_dim)

    def forward(self, x):
        # x = [batch, seq_len, *]
        x_flat = x.contiguous().view(-1, x.size(-1))
        out = self.proj_1(x_flat)  # [batch*seq_len, h_dim]
        out = F.relu(out)
        out = self.proj_2(out)     # [batch*seq_len, h_dim]
        out = out.view(-1, x.size(1), out.size(1))  # [batch, seq_len, h_dim]
        return out


class Prealign(nn.Module):
    def __init__(self, embed_dim, opt):  # embed_dim=300
        super(Prealign, self).__init__()
        self.hidden_size = opt['prealign_hidden_size']   # 280
        self.proj = nn.Linear(embed_dim, self.hidden_size, bias=False)
        self.output_size = opt['prealign_hidden_size']   # 280

    def forward(self, doc_emb, query_emb):  # doc_emb, query_emb
        doc_flat = doc_emb.contiguous().view(-1, doc_emb.size(-1))        # [batch * doc_len, 300]
        query_flat = query_emb.contiguous().view(-1, query_emb.size(-1))  # [batch * q_len, 300]

        doc_o = F.relu(self.proj(doc_flat)).view(doc_emb.size(0), doc_emb.size(1), -1) # [batch, doc_len, 280]
        query_o = F.relu(self.proj(query_flat)).view(query_emb.size(0), query_emb.size(1), -1)  # [batch, q_len, 280]

        score = doc_o.bmm(query_o.transpose(1, 2))          # [batch, doc_len, q_len]
        score = score.view(-1, score.size(-1))   # [batch*doc_len, q_len]
        score = F.softmax(score, 1).view(-1, doc_emb.size(1), query_emb.size(1))   # [batch, doc_len, q_len]

        out = score.bmm(query_o)   # [batch, doc_len, 280]
        return out
      

class LexiconEncoder(nn.Module):

    def create_word_embed(self, embedding, opt):
        vocab_size = opt['vocab_size']
        embed_dim = embedding.size(-1)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight.data = embedding
        fixed_embedding = embedding[opt['embed_tune_partial']:]
        self.register_buffer('fixed_embedding', fixed_embedding)
        self.fixed_embedding = fixed_embedding
        return embed_dim

    def create_cove(self, embedding, opt):
        self.ContextualEmbed = ContextualEmbed(opt['covec_path'], opt['vocab_size'], embedding)
        return self.ContextualEmbed.output_size      #600

    def create_prealign(self, embed_dim, opt): 
        self.prealign = Prealign(embed_dim, opt)
        return self.prealign.output_size 

    def create_ner_embed(self, opt):
        ner_vocab_size = opt['ner_vocab_size']
        ner_embed_dim = opt['ner_dim']
        self.ner_embedding = nn.Embedding(ner_vocab_size, ner_embed_dim, padding_idx=0)
        return ner_embed_dim

    def create_pos_embed(self, opt):
        pos_vocab_size = opt['pos_vocab_size']
        pos_embed_dim = opt['pos_dim']
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_embed_dim, padding_idx=0)
        return pos_embed_dim

    def __init__(self, opt, embedding):
        super(LexiconEncoder, self).__init__()

        # self.eval_embed; eval_embed.weight.data (model创建之后调用的)
        # self.embedding; self.embedding.weight.data; self.fixed_embedding; self.embedding_dim=300
        self.embedding_dim = self.create_word_embed(embedding, opt)
        # self.ContextualEmbed; self.covec_size=600
        self.covec_size = self.create_cove(embedding, opt) 
        # self.prealign; self.prealign_size=280
        self.prealign_size = self.create_prealign(self.embedding_dim, opt)
        # self.pos_embedding
        pos_size = self.create_pos_embed(opt)  # 9
        # self.ner_embedding
        ner_size = self.create_ner_embed(opt)  # 8
        match_feat_size = 3

        doc_hidden_size = self.embedding_dim + self.prealign_size + pos_size + ner_size + match_feat_size  # 300+280+9+8+3 = 600
        query_hidden_size = self.embedding_dim  # 300 

        self.doc_pwnn = PositionwiseNN(doc_hidden_size, opt['pwnn_hidden_size']) 
        self.query_pwnn = PositionwiseNN(query_hidden_size, opt['pwnn_hidden_size']) 
        self.doc_input_size = opt['pwnn_hidden_size']   # 128
        self.query_input_size = opt['pwnn_hidden_size']  # 128      
        
    def forward(self, batch):

        query_tok = Variable(batch['query_tok'])
        doc_tok = Variable(batch['doc_tok'])
        doc_pos = Variable(batch['doc_pos'])
        doc_ner = Variable(batch['doc_ner'])
        doc_fea = Variable(batch['doc_fea'][:,:,1:])

        # emb
        emb = self.embedding if self.training else self.eval_embed
        doc_emb = emb(doc_tok)
        query_emb = emb(query_tok)

        # cove
        doc_cove = self.ContextualEmbed(doc_tok)
        query_cove = self.ContextualEmbed(query_tok)

        # prealign
        q2d_attn = self.prealign(doc_emb, query_emb)
        doc_pos_emb = self.pos_embedding(doc_pos)
        doc_ner_emb = self.ner_embedding(doc_ner)

        doc_input = self.doc_pwnn(torch.cat([doc_emb, q2d_attn, doc_pos_emb, doc_ner_emb, doc_fea], 2))
        query_input = self.query_pwnn(query_emb)

        return doc_input, query_input, doc_cove, query_cove


class OneLayerBRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p):
        super(OneLayerBRNN, self).__init__()
        self.dropout_p = dropout_p
        self.hidden_size = hidden_size
        self.output_size = hidden_size        # 128
        self.rnn = nn.LSTM(input_size,
                           hidden_size,
                           num_layers=1,
                           batch_first=True, 
                           bidirectional=True)
        self.proj_1 = nn.Linear(hidden_size, hidden_size)
        self.proj_2 = nn.Linear(hidden_size, hidden_size)


    def forward(self, x):
        out, _ = self.rnn(x)   # [batch, seq_len, 128*2]
        out = out.view(-1, x.size(1), 2, self.hidden_size) 
        out_forward = out[:, :, 0, :].squeeze(2)    # [batch, seq_len, 128]
        out_forward = nn.Dropout(self.dropout_p)(out_forward)
        out_backward = out[:, :, 1, :].squeeze(2)   # [batch, seq_len, 128]
        out_backward = nn.Dropout(self.dropout_p)(out_backward)

        forward_proj = self.proj_1(out_forward).view(-1)
        backward_proj = self.proj_2(out_backward).view(-1)
        max_out = torch.max(forward_proj, backward_proj).view(x.size(0), x.size(1), self.hidden_size)   # [batch, seq_len, 128]
        return max_out


class Attention(nn.Module):
    def __init__(self, doc_input_size, query_input_size, output_size, dropout_p):
        super(Attention, self).__init__()
        self.proj_doc = nn.Linear(doc_input_size, output_size, bias=False)  # [256,128]
        self.proj_query = nn.Linear(query_input_size, output_size, bias=False)   # [256,128]
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, doc_hidden, query_hidden):
        doc_hidden_flat = doc_hidden.contiguous().view(-1, doc_hidden.size(-1))  # [batch*doc_len, 256]
        doc_o = F.relu(self.proj_doc(doc_hidden_flat))        # # [batch*doc_len, 128]
        doc_o = doc_o.view(doc_hidden.size(0), doc_hidden.size(1), -1)    # # [batch, doc_len, 128]

        query_hidden_flat = query_hidden.contiguous().view(-1, query_hidden.size(-1))   # [batch*query_len, 256]
        query_o = F.relu(self.proj_query(query_hidden_flat))   # [batch*query_len, 128]
        query_o = query_o.view(query_hidden.size(0), query_hidden.size(1), -1)     # [batch, query_len, 128]

        C = doc_o.bmm(query_o.transpose(1, 2))  # # [batch, doc_len, query_len]
        C = self.dropout(C)
        C = F.softmax(C.view(-1, C.size(2)), 1).view(-1, doc_hidden.size(1), query_hidden.size(1))   # [batch, doc_len, query_len]

        return C


# 计算U_p_hat
class Self_Attention(nn.Module):
    def __init__(self, dropout_p):
        super(Self_Attention, self).__init__()

    def forward(self, U_p):
        score = U_p.bmm(U_p.transpose(1, 2))   # [batch, doc_len, doc_len]

        diag_mask = torch.diag(score.data.new(score.size(1)).zero_()+1).byte().unsqueeze(0).expand_as(score)   # (batch, doc_len, doc_len)
        score.data.masked_fill_(diag_mask, -float('inf'))   # [batch, doc_len, doc_len]
        
        score = F.softmax(score.contiguous().view(-1, score.size(2)), 1).view(-1, U_p.size(1), U_p.size(1))    # [batch*doc_len, doc_len]
        
        U_p_hat = score.bmm(U_p)   # # [batch, doc_len, 128*4]
        return U_p_hat


# 计算s0
class Sum_Attention(nn.Module):
    def __init__(self, input_size):
        super(Sum_Attention, self).__init__()
        self.proj = nn.Linear(input_size, 1, bias=False)

    def forward(self, query_hidden):   # query_hidden = [batch, query_len, 128*2]
        query_hidden_flat = query_hidden.contiguous().view(-1, query_hidden.size(-1))
        query_o = self.proj(query_hidden_flat).view(query_hidden.size(0), query_hidden.size(1))   # [batch, query_len]
        
        score = F.softmax(query_o, 1).unsqueeze(1)  #    # [batch, 1, query_len]

        out = score.bmm(query_hidden).squeeze(1)    # [batch, 128*2]
        return out


def generate_mask(data, dropout_p, is_training):
    # data = (batch, 5)
    if not is_training:
        dropout_p = 0.0
    data = (1-dropout_p) * (data.zero_() + 1)
    for i in range(data.size(0)):  # 保证至少有一个不会被mask
        one = random.randint(0, data.size(1)-1)
        data[i][one] = 1
    mask = Variable(torch.bernoulli(data), requires_grad=False)
    return mask


class Answer(nn.Module):
    def __init__(self, query_input_size, mem_input_size, num_turn, dropout_p):
        super(Answer, self).__init__()
        self.num_turn = num_turn
        self.dropout_p = dropout_p

        self.proj_begin = nn.Linear(query_input_size, mem_input_size, bias=False)
        self.proj_end = nn.Linear(query_input_size+mem_input_size, mem_input_size, bias=False)
        self.proj_attn = nn.Linear(query_input_size, mem_input_size, bias=False)
        self.rnn = nn.GRU(mem_input_size,
                          query_input_size,
                          num_layers=1,
                          batch_first=True)
                          
        self.alpha = Parameter(torch.zeros(1, 1, 1))

    def forward(self, M, s0):
        # M = (batch, doc_len, 256)
        # s0 = (batch, 256)
        start_scores_list = []
        end_scores_list = []
        for _ in range(self.num_turn):
            start_scores = self.proj_begin(s0).unsqueeze(1).bmm(M.transpose(1, 2)).squeeze(1)    # (batch, n)
            start_scores_list.append(start_scores)

            end_input = F.softmax(start_scores, 1).unsqueeze(1).bmm(M).squeeze(1)  # (batch, 256)
            end_input = torch.cat([s0, end_input], 1)  # # (batch, 512)
            end_scores = self.proj_end(end_input).unsqueeze(1).bmm(M.transpose(1, 2)).squeeze(1)  # (batch,n)
            end_scores_list.append(end_scores)

            beta = F.softmax(self.proj_attn(s0).unsqueeze(1).bmm(M.transpose(1, 2)).squeeze(1), 1)  # (batch, n)
            x1 = beta.unsqueeze(1).bmm(M)  # x1 = (batch, 1, 256)
            _, s1 = self.rnn(x1, s0.unsqueeze(1).transpose(0, 1))  # s1 = (1, batch, 256)
            s0 = s1.transpose(0, 1).squeeze(1)   # (batch, 256)

        # random dropout  
        mask = generate_mask(self.alpha.data.new(M.size(0), self.num_turn), self.dropout_p, self.training) # (batch, 5)
        mask = [m.contiguous() for m in torch.unbind(mask, 1)]  # torch.unbind, 移除指定维后，返回一个元组, mask=(5,)
        start_scores_list = [mask[idx].view(M.size(0), 1).expand_as(inp)*F.softmax(inp, 1) 
                             for idx, inp in enumerate(start_scores_list)]  # inp=(batch, doc_len)
        start_scores = torch.stack(start_scores_list, 2)   #(batch, doc_len, 5)
        start_scores = torch.mean(start_scores, 2)   #(batch, doc_len)
        start_scores = torch.log(start_scores) 

        mask = generate_mask(self.alpha.data.new(M.size(0), self.num_turn), self.dropout_p, self.training) # (batch, 5)
        mask = [m.contiguous() for m in torch.unbind(mask, 1)]  # torch.unbind, 移除指定维后，返回一个元组, mask=(5,)
        end_scores_list = [mask[idx].view(M.size(0), 1).expand_as(inp)*F.softmax(inp, 1) 
                           for idx, inp in enumerate(end_scores_list)]        
        end_scores = torch.stack(end_scores_list, 2)     
        end_scores = torch.mean(end_scores, 2)   
        end_scores = torch.log(end_scores)

        return start_scores, end_scores


class SANQA_Network(nn.Module):
    def __init__(self, opt, embedding):
        super(SANQA_Network, self).__init__()

        ########## Lexicon Encoder 部分 
        self.lexicon_encoder = LexiconEncoder(opt, embedding=embedding)  
        query_input_size = self.lexicon_encoder.query_input_size   # 128  (PWNN之后的维度)
        doc_input_size = self.lexicon_encoder.doc_input_size       # 128  (PWNN之后的维度)
        covec_size = self.lexicon_encoder.covec_size               # 600

        ########## Contextual Encoder 部分（两层BiLSTM）
        self.doc_encoder_low = OneLayerBRNN(doc_input_size + covec_size, opt['contextual_hidden_size'], opt['dropout_p'])
        self.doc_encoder_high = OneLayerBRNN(self.doc_encoder_low.output_size + covec_size, opt['contextual_hidden_size'], opt['dropout_p'])
        doc_hidden_size = self.doc_encoder_low.output_size + self.doc_encoder_high.output_size         # 128*2=256

        self.query_encoder_low = OneLayerBRNN(query_input_size + covec_size, opt['contextual_hidden_size'], opt['dropout_p'])
        self.query_encoder_high = OneLayerBRNN(self.query_encoder_low.output_size + covec_size, opt['contextual_hidden_size'], opt['dropout_p'])
        query_hidden_size = self.query_encoder_low.output_size + self.query_encoder_high.output_size   # 128*2=256

        ########## Memory Generation 部分 
        # 输入：doc_hidden = [batch, doc_len, doc_hidden_size]
        # 输入：query_hidden = [batch, query_len, query_hidden_size]
        # 输出：C=[batch, doc_len, query_len]
        self.atten = Attention(doc_hidden_size, query_hidden_size, opt['atten_hidden_size'], opt['dropout_p'])

        # 输入：U_p =  [batch, doc_len, doc_hidden_size+query_hidden_size]
        # 输出：U_p_hat = [batch, doc_len, doc_hidden_size+query_hidden_size]
        self.self_atten = Self_Attention(opt['dropout_p'])

        # 输入：[U_p; U_p_hat]
        # 输出：M = [batch, doc_len, 2*opt['contextual_hidden_size']]
        self.mem_rnn = nn.LSTM((doc_hidden_size+query_hidden_size)*2,
                               opt['contextual_hidden_size'],
                               num_layers=1,
                               batch_first=True, 
                               bidirectional=True)

        # 输入：query_hidden = [batch, query_len, query_hidden_size]
        # 输出：s0 = [batch, query_hidden_size]
        self.query_sum_attn = Sum_Attention(query_hidden_size)

        # 输入：M = [batch, doc_len, 2*opt['contextual_hidden_size']]
        # 输出：M_sum_attn = [batch, 2*opt['contextual_hidden_size']]
        self.mem_sum_attn = Sum_Attention(opt['contextual_hidden_size']*2)

        ########## Answer 部分 
        # 输入：s0 = [batch, query_hidden_size]
        # 输入：M_sum_attn = [batch, 2*opt['contextual_hidden_size']]
        self.decoder = Answer(query_hidden_size, opt['contextual_hidden_size']*2, opt['answer_num_turn'], opt['dropout_p']) 

        # 输入：[s0; M_sum_attn] = [batch, query_hidden_size+2*opt['contextual_hidden_size']]
        self.answable_classifier = nn.Linear(query_hidden_size+opt['contextual_hidden_size']*2, opt['label_size'])  


    def forward(self, batch):

        ####### Lexicon Encoder 部分 
        doc_input, query_input, doc_cove, query_cove = self.lexicon_encoder(batch)

        ####### contextual encoder 部分（两层BiLSTM）
        doc_low = self.doc_encoder_low(torch.cat([doc_input, doc_cove], 2)) 
        doc_high = self.doc_encoder_high(torch.cat([doc_low, doc_cove], 2)) 
        doc_hidden = torch.cat([doc_low, doc_high], 2)   # [batch, doc_len, 128*2]

        query_low = self.query_encoder_low(torch.cat([query_input, query_cove], 2)) 
        query_high = self.query_encoder_high(torch.cat([query_low, query_cove], 2)) 
        query_hidden = torch.cat([query_low, query_high], 2)    # [batch, query_len, 128*2]

        ####### Memory Generation 部分 
        C = self.atten(doc_hidden, query_hidden)   # [batch, doc_len, query_len]
        U_p = torch.cat([doc_hidden, C.bmm(query_hidden)], 2)   # [batch, doc_len, 128*4]
        U_p_hat = self.self_atten(U_p)         # [batch, doc_len, 128*4]
        M, _ = self.mem_rnn(torch.cat([U_p, U_p_hat], 2))  # [batch, doc_len, 128*2]

        s0 = self.query_sum_attn(query_hidden)   # [batch, 128*2]
        M_sum_attn = self.mem_sum_attn(M)        # [batch, 128*2]
  
        ####### Answer 部分 
        start_scores, end_scores = self.decoder(M, s0)
        pred_scores = F.sigmoid(self.answable_classifier(torch.cat([M_sum_attn, s0], 1)))

        return start_scores, end_scores, pred_scores
        