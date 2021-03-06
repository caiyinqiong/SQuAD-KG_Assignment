import argparse
from datetime import datetime
import os
import pickle
import torch
import json
import tqdm
from my_utils.train_utils import BatchGen, predict_squda
from sanQA.sanQA_model import SANQA_Model
from my_utils.evaluate import my_evaluate

# 返回embedding矩阵 和 更新参数中三个词典的长度
def load_meta(opt, squda_meta_data_path):
    with open(squda_meta_data_path, 'rb') as f:
        meta_data = pickle.load(f)
    embedding = torch.Tensor(meta_data['embedding'])
    opt['vocab_size'] = len(meta_data['vocab'])
    opt['pos_vocab_size'] = len(meta_data['vocab_tag'])
    opt['ner_vocab_size'] = len(meta_data['vocab_ner'])
    return embedding, opt


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev_data_path', default = 'F:\\0recentwork\\SQuDA\\data\\dev-v2.0.json')    # /home/caiyinqiong/squda/data/dev-v2.0.json
    parser.add_argument('--squda_meta_data_path', default = 'F:\\0recentwork\\SQuDA\\output\\squda_v2.pick')   # /home/caiyinqiong/squda/output-SAN
    parser.add_argument('--train_propress_data_path', default='F:\\0recentwork\\SQuDA\\output\\train_preprocess.json')  # /home/caiyinqiong/squda/output-SAN
    parser.add_argument('--dev_propress_data_path', default='F:\\0recentwork\\SQuDA\\output\\dev_preprocess.json')  # /home/caiyinqiong/squda/output-SAN
    parser.add_argument('--covec_path', default='F:\\0recentwork\\SQuDA\\data\\wmtlstm-b142a7f2.pth')  # /home/caiyinqiong/squda/data/wmtlstm-b142a7f2.pth
    parser.add_argument('--out_model_dir', default='F:\\0recentwork\\SQuDA\\output_sanQA\\model')   ## 
    
    parser.add_argument('--vocab_size', default=1000)
    parser.add_argument('--pos_vocab_size', default=10)
    parser.add_argument('--ner_vocab_size', default=10)

    # model
    parser.add_argument('--batch_size', default=32) ######### 原文设32
    parser.add_argument('--epochs', default=10)     #######################################
    parser.add_argument('--learning_rate', default=0.002)  ### 原文设置：初始0.002，每10 epoch 降低一半
    parser.add_argument('--embed_tune_partial', default=1000)  ## 只调整前1000个词的embedding
    parser.add_argument('--max_len', default=15)

    ## network
    parser.add_argument('--dropout_p', type=float, default=0.4)  ## 用于 LSTM 和 Answer
    # lexicon
    parser.add_argument('--pwnn_hidden_size', default=128)  ## PositionWiseNN的隐单元数
    parser.add_argument('--pos_dim', default=9)  ## 词性标注的特征向量长度
    parser.add_argument('--ner_dim', default=8)    ## 实体的特征向量长度
    parser.add_argument('--prealign_hidden_size', default=280)
    # contextual
    parser.add_argument('--contextual_hidden_size', default=128)   # 两层 BiLSTM的隐单元数
    # memory
    parser.add_argument('--atten_hidden_size', default=128)   # H_p -> H_p_hat
    # answer
    parser.add_argument('--answer_num_turn', default=5)
    parser.add_argument('--label_size', default=1)
  
    args = parser.parse_args()
    return args


def main():
    args = set_args()
    opt = vars(args)

    print('start load data...')
    embedding, opt = load_meta(opt, args.squda_meta_data_path)
    train_data = BatchGen(args.train_propress_data_path, args.batch_size, is_train=True)
    dev_data = BatchGen(args.dev_propress_data_path, args.batch_size, is_train=False)
    print('load data end!')

    ## model define
    # param = torch.load('F:\\0recentwork\\SQuDA\\output\\model\\model_chechpoint_epoch_0.pt')
    # opt = param['config']
    # embedding = param['state_dict']['network']['lexicon_encoder.embedding.weight']
    # model = My_Model(opt, embedding)
    # model.network.load_state_dict = param['state_dict']['network']  
    model = SANQA_Model(opt, embedding)
    print(model.network)
    model.setup_eval_embed(embedding)  #### DocReaderModel中的方法 
    print('模型总参数量：' + str(model.total_param))

    for epoch in range(args.epochs):
        print('epoch ' + str(epoch))
        train_data.reset()   ####  BatchGen中的方法，打乱batch的顺序
        start = datetime.now()
    
        ## train
        print('第 ' + str(epoch) + ' 轮的训练开始......')
        for i, batch in enumerate(train_data):            
            model.update(batch)  #### DocReaderModel中的方法
            #break  ########################################################### debug
            remain_time = str((datetime.now() - start) / (i + 1) * (len(train_data) - i - 1)).split('.')[0]
            print('loss = {}, remain_time = {}'.format(model.train_loss.avg, remain_time))
        print('第 ' + str(epoch) + ' 轮的训练结束！ train_loss = ' + str(model.train_loss.avg))

        ## save
        model_out_file_path = os.path.join(args.out_model_dir, 'model_chechpoint_epoch_{}.pt'.format(epoch))
        model.save(model_out_file_path)  

        ## predict
        print('第 ' + str(epoch) + ' 轮的预测开始......')
        answer_pred_results = predict_squda(model, dev_data)   ## 预测的结果：{uid:answer_text}
        predict_out_path = os.path.join(args.out_model_dir, 'dev_predict_out_{}.json'.format(epoch))
        with open(predict_out_path, 'w') as f:   
            json.dump(answer_pred_results, f)
        print('第 ' + str(epoch) + ' 轮的预测结束！')
        exact_score, f1_score = my_evaluate(opt['dev_data_path'], answer_pred_results)
        print('第 ' + str(epoch) + ' 轮得分：exact_score=' + str(exact_score) + ', f1_score=' + str(f1_score))
       
        print('epoch ' + str(epoch) + ' 结束！')
    print('train done!')
    

if __name__ == "__main__":
    main()
