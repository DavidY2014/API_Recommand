import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
import random
import sys
import math
parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules
from helper import PAD_ID
from modules import CharEmbedding, RNNEncoder, RNNDecoder

    
class RNNEncDec(nn.Module):
    '''The basic Hierarchical Recurrent Encoder-Decoder model. '''
    def __init__(self, config):
        super(RNNEncDec, self).__init__()
        self.vocab_size = config['vocab_size'] 
        self.maxlen=config['max_sent_len']
        self.temp=config['temp']

        #下面是原来的RNN结构
        self.desc_embedder = nn.Embedding(self.vocab_size, config['emb_dim'], padding_idx=PAD_ID)
        self.api_embedder = nn.Embedding(self.vocab_size, config['emb_dim'], padding_idx=PAD_ID)
                                                        
        self.encoder = RNNEncoder(self.desc_embedder, None, config['emb_dim'], config['n_hidden'],
                        True, config['n_layers'], config['noise_radius']) # utter encoder: encode response to vector
        self.ctx2dec = nn.Sequential( # from context to decoder initial hidden
            nn.Linear(2*config['n_hidden'], config['n_hidden']),
            nn.Tanh(),
        )
        self.ctx2dec.apply(self.init_weights)
        self.decoder = RNNDecoder(self.api_embedder, config['emb_dim'], config['n_hidden'],
                               self.vocab_size, config['attention'], 1, config['dropout']) # decoder: P(x|c,z)


        
    def init_weights(self, m):# Initialize Linear Weight for GAN
        if isinstance(m, nn.Linear):        
            m.weight.data.uniform_(-0.08, 0.08)#nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.)   
        
    def forward(self, src_seqs, src_lens, target, tar_lens):
        #下面是原来的RNN
        #接受输入，但是是batchsize的src_seqs
        c, hids = self.encoder(src_seqs, src_lens)#c = [3,2000],hids [3,50,2000]
        init_h, hids = self.ctx2dec(c), self.ctx2dec(hids) # init_h = [3,1000],hids = [3,50,1000]
        src_pad_mask = src_seqs.eq(PAD_ID) # 如果元素为pad_id =0 ，那么mask矩阵就是True
        output,_ = self.decoder(init_h, hids, src_pad_mask, None, target[:,:-1], (tar_lens-1))
                                             # decode from z, c  # output: [batch x seq_len x n_tokens] [3,49,10000]
    
    
#         print(target.shape) #[3,50]
        dec_target = target[:,1:].clone() # [3,49]
        dec_target[target[:,1:]==PAD_ID]=-100 # 对矩阵等于pad_id的进行填充为-100,这样在计算后可以降低影响
#         print(output.view(-1, self.vocab_size).shape) #[147,10000]
#         print(dec_target.view(-1).shape)  #[147]
        loss = nn.CrossEntropyLoss()(output.view(-1, self.vocab_size)/self.temp, dec_target.view(-1))
        return loss

    def valid(self, src_seqs, src_lens, target, tar_lens):
        self.eval()       
        loss = self.forward(src_seqs, src_lens, target, tar_lens)
        return {'valid_loss': loss.item()}
        
    def sample(self, src_seqs, src_lens, n_samples, decode_mode='beamsearch'):    
        self.eval()
        src_pad_mask = src_seqs.eq(PAD_ID)
        c, hids = self.encoder(src_seqs, src_lens)
        init_h, hids = self.ctx2dec(c), self.ctx2dec(hids)
        if decode_mode =='beamsearch':
            sample_words, sample_lens, _ = self.decoder.beam_decode(init_h, hids, src_pad_mask, None, 12, self.maxlen, n_samples)
                                                                   #[batch_size x n_samples x seq_len]
            sample_words, sample_lens = sample_words[0], sample_lens[0]
        else:
            sample_words, sample_lens = self.decoder.sampling(init_h, hids, src_pad_mask, None, self.maxlen, decode_mode)  
        return sample_words, sample_lens   
    
    def adjust_lr(self):
        #self.lr_scheduler_AE.step()
        return None


