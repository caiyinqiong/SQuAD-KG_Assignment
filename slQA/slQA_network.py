import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from allennlp.modules.elmo import Elmo

class Encoder(nn.Module):
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

    def create_word_embed(self, opt, embedding):
        vocab_size = opt['vocab_size']
        embed_dim = 300
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight.data = embedding
        fixed_embedding = embedding[opt['embed_tune_partial']:]
        self.register_buffer('fixed_embedding', fixed_embedding)
        self.fixed_embedding = fixed_embedding
        return embed_dim

    def create_elmo(self, opt):
        self.elmo = Elmo(opt['elmo_config_path'], opt['elmo_weight_path'], num_output_representations=3)
        return self.elmo.get_output_dim()    

    def __init__(self, opt, embedding):
        super(Encoder, self).__init__()
        self.dropout_p = opt['dropout_p']

        # self.eval_embed; eval_embed.weight.data (model创建之后调用的)
        # self.embedding; self.embedding.weight.data, self.fixed_embedding, self.embedding_dim=300
        self.embedding_dim = self.create_word_embed(opt, embedding) 
        # self.elmo， self.elmo_size=1024
        self.elmo_size = self.create_elmo(opt) 

        self.lstm = nn.LSTM(self.embedding_dim+self.elmo_size,   # 300+1024
                            opt['encoder_lstm_hidden_size'],     # 128
                            num_layers=1,
                            bidirectional=True,
                            batch_first=True)
        
        self.output_size = 2 * opt['encoder_lstm_hidden_size'] + self.elmo_size   # 128*2 +1024 = 1280

        # 手工特征
        pos_size = self.create_pos_embed(opt)  # self.pos_embedding, 18
        ner_size = self.create_ner_embed(opt)  # self.ner_embedding, 18
        feat_size = 4
        self.manual_fea_size = pos_size + ner_size + feat_size    # 40        

    def forward(self, batch):
        doc_tok = Variable(batch['doc_tok'])
        doc_ctok = Variable(batch['doc_ctok'])
        doc_pos = Variable(batch['doc_pos'])
        doc_ner = Variable(batch['doc_ner'])
        doc_fea = Variable(batch['doc_fea'])
        query_tok = Variable(batch['query_tok'])
        query_ctok = Variable(batch['query_ctok'])
        
        emb = self.embedding if self.training else self.eval_embed
        doc_emb = emb(doc_tok)
        query_emb = emb(query_tok)

        doc_elmo = self.elmo(doc_ctok)['elmo_representations'][0]
        query_elmo = self.elmo(query_ctok)['elmo_representations'][0]

        doc_o, _ = self.lstm(torch.cat([doc_emb, doc_elmo], 2))   # [batch, seq_len, 200]
        doc_o = nn.Dropout(self.dropout_p)(doc_o)
        U_P = torch.cat([doc_o, doc_elmo], 2)   # [batch, seq_len, 200+1024]

        query_o, _ = self.lstm(torch.cat([query_emb, query_elmo], 2))
        query_o = nn.Dropout(self.dropout_p)(query_o)
        U_Q = torch.cat([query_o, query_elmo], 2)   

        doc_pos_emb = self.pos_embedding(doc_pos)
        doc_ner_emb = self.ner_embedding(doc_ner)
        doc_manual_feature = torch.cat([doc_pos_emb, doc_ner_emb, doc_fea], -1)

        return U_Q, U_P, doc_manual_feature


class Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Attention, self).__init__()
        self.proj_1 = nn.Linear(input_size, hidden_size, bias=False)
        self.proj_2 = nn.Linear(input_size, hidden_size, bias=False)
    
    def forward(self, U_Q, U_P):
        U_Q_o = F.relu(self.proj_1(U_Q))  # [batch, query_len, hidden_size]
        U_P_o = F.relu(self.proj_2(U_P))     # [batch, doc_len, hidden_size]

        S = U_Q_o.bmm(U_P_o.transpose(1, 2))  # # [batch, query_len, doc_len]
        return S


class Fuse(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Fuse, self).__init__()
        self.proj = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.gate = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y_hat):
        m = self.tanh(self.proj(torch.cat([x, y_hat, x * y_hat, x - y_hat], dim=-1)))
        gate = self.sigmoid(self.gate(torch.cat([x, y_hat, x * y_hat, x - y_hat], dim=-1)))
        return gate * m + (1 - gate) * x


class Proj_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p):
        super(Proj_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.dropout_p = dropout_p
    def forward(self, x):
        out, _ = self.lstm(x)
        out = nn.Dropout(self.dropout_p)(out)
        return out


class Doc_Self_Atten(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Doc_Self_Atten, self).__init__()
        self.proj = nn.Linear(input_size, hidden_size, bias=False)

    def forward(self, D):
        L = self.proj(D).bmm(D.transpose(1, 2))
        L = F.softmax(L.contiguous().view(-1, L.size(2)), dim=-1).view(-1, L.size(1), L.size(2))   # [batch, doc_len, doc_len]
        
        D_widetilde = L.bmm(D)   # [batch, doc_len, 200]
        return D_widetilde


class Self_Atten(nn.Module):
    def __init__(self, input_size):
        super(Self_Atten, self).__init__()
        self.proj = nn.Linear(input_size, 1, bias=False)

    def forward(self, Q_prime2):   # Q_prime2 = [batch, query_len, input_size=200]
        gramma = F.softmax(self.proj(Q_prime2).squeeze(2), dim=1)  # [batch, query_len],行求softmax
        q = gramma.unsqueeze(1).bmm(Q_prime2).squeeze(1)     # [batch, 200]
        return q


class Bi_Linear(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Bi_Linear, self).__init__()
        self.proj = nn.Linear(input_size, hidden_size, bias=False)

    def forward(self, q, D_prime2):
        out = self.proj(q).unsqueeze(1).bmm(D_prime2.transpose(1, 2)).squeeze(1)    # [batch,n]
        return out


class SLQA_Network(nn.Module):
    def __init__(self, opt, embedding):
        super(SLQA_Network, self).__init__()

        # Encoder Layer
        self.encoder = Encoder(opt, embedding)   
        encoder_out_size = self.encoder.output_size   # 200+1024=1224
        manual_fea_size = self.encoder.manual_fea_size   # 40


        # Attention Layer
        self.atten = Attention(encoder_out_size, opt['atten_hidden_size'])
        self.fuse_doc = Fuse(4*encoder_out_size, encoder_out_size)
        self.fuse_query = Fuse(4*encoder_out_size, encoder_out_size)

        self.doc_proj_lstm = Proj_LSTM(encoder_out_size+manual_fea_size, opt['atten_lstm_hidden_size'], opt['dropout_p'])
        self.doc_self_atten = Doc_Self_Atten(2*opt['atten_lstm_hidden_size'], 2*opt['atten_lstm_hidden_size'])
        self.doc_fuse = Fuse(4*2*opt['atten_lstm_hidden_size'], 2*opt['atten_lstm_hidden_size'])
        self.doc_proj_lstm2 = Proj_LSTM(2*opt['atten_lstm_hidden_size'], opt['atten_lstm_hidden_size'], opt['dropout_p'])
        self.doc_sum_atten = Self_Atten(2*opt['atten_lstm_hidden_size'])   ## 为了判断问题是否可回答用

        self.query_proj_lstm = Proj_LSTM(encoder_out_size, opt['atten_lstm_hidden_size'], opt['dropout_p'])
        self.query_self_atten = Self_Atten(2*opt['atten_lstm_hidden_size'])


        # Model & Output Layer
        self.start_output_layer = Bi_Linear(2*opt['atten_lstm_hidden_size'], 2*opt['atten_lstm_hidden_size']) 
        self.end_output_layer = Bi_Linear(2*opt['atten_lstm_hidden_size'], 2*opt['atten_lstm_hidden_size']) 
        self.classifier = nn.Linear(2*opt['atten_lstm_hidden_size']+2*opt['atten_lstm_hidden_size'], opt['label_size'])

        
    def forward(self, batch):

        ############# Encoder Layer
        # Q = [batch, query_len, encoder_out_size=1224]
        # P = [batch, doc_len, encoder_out_size=1224]
        # doc_manual_feature = [batch, doc_len, 18+18+4=40]
        Q, P, doc_manual_feature = self.encoder(batch) 


        ################## Attention Layer
        # [batch, query_len, doc_len]
        S = self.atten(Q, P)  
        # P2Q atten (对S的列求softmax)
        alpha = F.softmax(S.contiguous().view(S.size(1), -1), dim=0).view(-1, S.size(1), S.size(2))  # [batch, query_len, doc_len]
        Q_hat = alpha.transpose(1,2).bmm(Q)   # # [batch, doc_len, encoder_out_size=1280]
        # Q2P atten (对S的行求softmax)
        beta = F.softmax(S.contiguous().view(-1, S.size(-1)), dim=1).view(-1, S.size(1), S.size(2))  # [batch, query_len, doc_len]
        P_hat = beta.bmm(P)    # [batch, query_len, encoder_out_size=1280]
        # fuse
        P_prime = self.fuse_doc(P, Q_hat)     # [batch, doc_len, encoder_out_size=1224]
        Q_prime = self.fuse_query(Q, P_hat)   # [batch, query_len, encoder_out_size=1224]

        # doc self-atten + fuse
        D = self.doc_proj_lstm(torch.cat([P_prime, doc_manual_feature], dim=-1))   # [batch, doc_len, 200]
        D_widetilde = self.doc_self_atten(D)      # [batch, doc_len, 200]
        D_prime = self.doc_fuse(D, D_widetilde)   # [batch, doc_len, 200]
        D_prime2 = self.doc_proj_lstm2(D_prime)   # [batch, doc_len, 200]
        d = self.doc_sum_atten(D_prime2)          ## 为了分类用
        # query self-atten + fuse
        Q_prime2 = self.query_proj_lstm(Q_prime)  # [batch, query_len, 200]
        q = self.query_self_atten(Q_prime2)  # [batch, 200]

 
        ############## Model & Output Layer
        start_scores = self.start_output_layer(q, D_prime2)
        end_scores = self.end_output_layer(q, D_prime2)
        label_scores = F.sigmoid(self.classifier(torch.cat([q, d], dim=-1)))

        
        return start_scores, end_scores, label_scores
