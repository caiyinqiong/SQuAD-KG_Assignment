import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Prealign(nn.Module):
    def __init__(self, embed_dim, opt):
        super(Prealign,self).__init__()
        self.hidden_size = opt['prealign_hidden_size']   # 128
        self.proj = nn.Linear(embed_dim, self.hidden_size)  

    def forward(self, doc_embed, query_embed):
        doc_embed_flat = doc_embed.contiguous().view(-1, doc_embed.size(-1))
        doc_embed_o = self.proj(doc_embed_flat)
        doc_embed_o = F.relu(doc_embed_o).view(doc_embed.size(0), doc_embed.size(1), -1)

        query_embed_flat = query_embed.contiguous().view(-1, query_embed.size(-1))
        query_embed_o = self.proj(query_embed_flat)
        query_embed_o = F.relu(query_embed_o).view(query_embed.size(0), query_embed.size(1), -1)

        scores = doc_embed_o.bmm(query_embed_o.transpose(1, 2))  # [batch, doc_len, query_len]
        scores = F.softmax(scores.view(-1, query_embed.size(1)), 1)
        scores = scores.view(-1, doc_embed.size(1), query_embed.size(1))

        out = scores.bmm(query_embed)
        return out


class LexiconEncoder(nn.Module):

    def create_word_embed(self, embedding, opt):
        vocab_size = opt['vocab_size']
        embed_dim = embedding.size(-1)   # 300
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight.data = embedding
        fixed_embedding = embedding[opt['embed_tune_partial']:]
        self.register_buffer('fixed_embedding', fixed_embedding)
        self.fixed_embedding = fixed_embedding
        return embed_dim

    def create_prealign(self, embed_dim, opt): 
        self.prealign = Prealign(embed_dim, opt) 

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
        self.dropout = nn.Dropout(opt['dropout_p'])

        # self.eval_embed; eval_embed.weight.data  (创建完model时就设置了)
        # self.embedding; self.embedding.weight.data;self.fixed_embedding
        self.embedding_dim = self.create_word_embed(embedding, opt)  # 300
        self.query_output_size = self.embedding_dim
        self.doc_output_size = self.embedding_dim

        # self.prealign     
        self.create_prealign(self.embedding_dim, opt)
        self.prealign_size = self.embedding_dim
        self.doc_output_size += self.prealign_size     # +300

        # self.ner_embedding
        pos_size = self.create_pos_embed(opt)
        self.doc_output_size += pos_size              # +8

        # self.pos_embedding
        ner_size = self.create_ner_embed(opt)
        self.doc_output_size += ner_size              # +8

        feat_size = 4
        self.doc_output_size += feat_size
            
    def forward(self, batch):    
        query_tok = Variable(batch['query_tok'])
        doc_tok = Variable(batch['doc_tok'])
        doc_pos = Variable(batch['doc_pos'])
        doc_ner = Variable(batch['doc_ner'])
        doc_fea = Variable(batch['doc_fea'])
     
        emb = self.embedding if self.training else self.eval_embed      
        doc_emb = emb(doc_tok)
        query_emb = emb(query_tok)
        if self.training:
            doc_emb = self.dropout(doc_emb)
            query_emb = self.dropout(query_emb)

        prealign = self.prealign(doc_emb, query_emb)       
        doc_ner_emb = self.ner_embedding(doc_ner)
        doc_pos_emb = self.pos_embedding(doc_pos)
        doc_fea = doc_fea

        query_out = query_emb
        doc_out = torch.cat([doc_emb, prealign, doc_ner_emb, doc_pos_emb, doc_fea], -1)
 
        return query_out, doc_out


class Doc_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p):
        super(Doc_Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = hidden_size * 6   # 三层双向
        self.dropout_p = dropout_p

        self.low = nn.LSTM(self.input_size,
                           self.hidden_size,
                           num_layers=1,
                           batch_first=True, 
                           bidirectional=True)
        self.middle = nn.LSTM(self.hidden_size * 2,
                           self.hidden_size,
                           num_layers=1,
                           batch_first=True, 
                           bidirectional=True)
        self.high = nn.LSTM(self.hidden_size * 2,
                           self.hidden_size,
                           num_layers=1,
                           batch_first=True, 
                           bidirectional=True)
    def forward(self, doc_feature):
        # doc_feature = [batch, doc_len, 620]
        out_low, _ = self.low(doc_feature)    # [batch, doc_len, 128*2]
        out_low = nn.Dropout(self.dropout_p)(out_low)
        out_middle, _ = self.middle(out_low)  # [batch, doc_len, 128*2]
        out_middle = nn.Dropout(self.dropout_p)(out_middle)
        out_high, _ = self.high(out_middle)   # [batch, doc_len, 128*2]

        out = torch.cat([out_low, out_middle, out_high], -1)   # [batch, doc_len, 128*6]
        return out


class Self_Attention(nn.Module):
    def __init__(self, input_size):
        super(Self_Attention, self).__init__()
        self.linear = nn.Linear(input_size, 1, bias=False)

    def forward(self, encoding):
        # encoding = [batch, seq_len, *]
        encoding_flat = encoding.contiguous().view(-1, encoding.size(-1)) # [batch*seq_len, *]
        scores = self.linear(encoding_flat)    # [batch*seq_len, 1]
        scores = scores.view(encoding.size(0), encoding.size(1))   # [batch, seq_len]
        scores = F.softmax(scores, 1)   # [batch, seq_len]
        
        out = scores.unsqueeze(1).bmm(encoding).squeeze(1)    #  [batch, *]
        return out


class Answer(nn.Module):
    def __init__(self, input_size, output_size):
        super(Answer, self).__init__()
        self.start_proj = nn.Linear(input_size, output_size, bias=False)
        self.end_proj = nn.Linear(input_size, output_size, bias=False)

    def forward(self, doc, query_sum_attn):
        # query_sum_attn = [batch, 128*2]
        # doc = [batch, doc_len, 128*6]
        start_score = self.start_proj(query_sum_attn).unsqueeze(2)   # [batch, 128*6, 1]
        start_score = doc.bmm(start_score).squeeze(2)  # [batch, doc_len]

        end_score = self.end_proj(query_sum_attn).unsqueeze(2)
        end_score = doc.bmm(end_score).squeeze(2)

        return start_score, end_score


class DocQA_Network(nn.Module):
    def __init__(self, opt, embedding):
        super(DocQA_Network, self).__init__()

        self.lexicon_encoder = LexiconEncoder(opt, embedding=embedding) 
        query_input_size = self.lexicon_encoder.query_output_size     # 300
        doc_input_size = self.lexicon_encoder.doc_output_size       # 300 + 300 + 8 + 8 +4 = 620


        self.doc_encoder = Doc_Encoder(doc_input_size, opt['contextual_hidden_size'], opt['dropout_p'])
        doc_hidden_size = self.doc_encoder.output_size     # 128 * 6

        self.query_encoder = nn.LSTM(query_input_size,
                                     opt['contextual_hidden_size'],
                                     num_layers=3,
                                     batch_first=True,
                                     dropout=opt['dropout_p'],
                                     bidirectional=True)
        query_hidden_size = opt['contextual_hidden_size'] * 2    # 128 * 2


        # 输入(batch, query_len, 128*2)
        # 输出(batch, 128*2)
        self.query_sum_attn = Self_Attention(query_hidden_size)

        # 输入(batch, doc_len, 128 * 6)
        # 输出(batch, 128 * 6)
        self.doc_sum_attn = Self_Attention(doc_hidden_size)


        self.decoder = Answer(query_hidden_size, doc_hidden_size) 
        self.classifier = nn.Linear(doc_hidden_size+query_hidden_size, opt['label_size'])


    def forward(self, batch):
        
        # doc_feature = [batch, doc_len, 620]
        # query_feature = [batch, query_len, 300]
        query_feature, doc_feature = self.lexicon_encoder(batch)

        # [batch, doc_len, 128*6]
        doc_encoding = self.doc_encoder(doc_feature)
        # [batch, query_len, 128*2]
        query_encoding, _ = self.query_encoder(query_feature)  

        # [batch, 128*2]
        query_sum = self.query_sum_attn(query_encoding)  
        # [batch, 128*6]
        doc_sum = self.doc_sum_attn(doc_encoding)
  
        # start_scores = end_scores = [batch, doc_len]
        start_scores, end_scores = self.decoder(doc_encoding, query_sum)
        # [batch, 1]
        pred_scores = F.sigmoid(self.classifier(torch.cat([doc_sum, query_sum], -1)))

        return start_scores, end_scores, pred_scores
        