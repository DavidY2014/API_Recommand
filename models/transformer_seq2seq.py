import torch
import torch.nn as nn
import copy
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
from modules import MultiHeadedAttention, PositionwiseFeedForward, PositionalEncoding,EncoderDecoder,Encoder,EncoderLayer,Decoder,DecoderLayer,Embeddings,Generator,subsequent_mask
from torch.autograd import Variable


class Transformer_EncoderDecoder(nn.Module):
    """
    标准的Encoder-Decoder架构
    """
    def __init__(self, config):
        super(Transformer_EncoderDecoder, self).__init__()
        c = copy.deepcopy
        self.attn = MultiHeadedAttention(config['head'], config['emb_dim'])
        self.ff = PositionwiseFeedForward(config['emb_dim'], config['d_ff'], config['drop_out'])
        self.position = PositionalEncoding(config['emb_dim'], config['drop_out'])
        self.encoder = Encoder(EncoderLayer(config['emb_dim'], c(self.attn), c(self.ff), config['drop_out']), config['N_layers'])
        self.decoder = Decoder(DecoderLayer(config['emb_dim'], c(self.attn), c(self.attn), c(self.ff), config['drop_out']), config['N_layers'])
        self.src_embed = nn.Sequential(Embeddings(config['emb_dim'], config['vocab_size']), c(self.position))
        self.tgt_embed = nn.Sequential(Embeddings(config['emb_dim'], config['vocab_size']), c(self.position))
        self.generator = Generator(config['emb_dim'], config['vocab_size'])
        self.fc_out = nn.Linear(config['emb_dim'], config['vocab_size'])
        
        self.model = EncoderDecoder(
            self.encoder,
            self.decoder,
            self.src_embed,
            self.tgt_embed,
            self.generator)



        # forward函数调用自身encode方法实现encoder，然后调用decode方式实现decoder
    def forward(self, src, src_lens, tgt, tar_lens,pad=0):
        #src的shape=tgt的shape：[batch_size,max_length]
        "Take in and process masked src and target sequences."
        # 随机初始化参数，这非常重要
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

        #生成mask
        #src：[batch_size,max_legth]
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            #self.trg表示去掉每行的最后一个单词=====》相当于t-1时刻
            self.tgt = tgt[:, :-1]
            
            #self.trg_y表示去掉每行的第一个单词=====》相当于t时刻
            #decode 就是使用encoder和t-1时刻去预测t时刻
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = \
                self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()
        
        #模型workflow
        output = self.decode(self.encode(src, self.src_mask), self.src_mask,
                            tgt[:,:-1], self.tgt_mask)
        
        output = self.fc_out(output)   #[3,49,10000]  
        
        dec_tgt = tgt[:,1:].clone() # [3,49]
        dec_tgt[tgt[:,1:]==PAD_ID]=-100 # 对矩阵等于pad_id的进行填充为-100,这样在计算后可以降低影响
            
        loss = nn.CrossEntropyLoss()(output.view(-1, 10000)/1.0, dec_tgt.view(-1))
        return loss

    def init_weights(self, m):# Initialize Linear Weight for GAN
        if isinstance(m, nn.Linear):        
            m.weight.data.uniform_(-0.08, 0.08)#nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.)   
    
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
    
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "创建Mask，使得我们不能attend to未来的词"
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask  
    

    
    
#pytorch 自带的transformer模块,暂时不用
class Transformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_len,
        device,
    ):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx
        # (N, src_len)
        return src_mask.to(self.device)

    def forward(self, src, src_lens, trg, tar_lens):
        print(src.shape)
        
        src_positions = (
            torch.arange(0, src_seq_length)
            .unsqueeze(1)
            .expand(src_seq_length, N)
            .to(self.device)
        )

        trg_positions = (
            torch.arange(0, trg_seq_length)
            .unsqueeze(1)
            .expand(trg_seq_length, N)
            .to(self.device)
        )

        embed_src = self.dropout(
            (self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        )
        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
        )

        src_padding_mask = self.make_src_mask(src).to(self.device)
        trg_mask = self.transformer.generate_square_subsequent_mask(tar_lens).to(
            self.device
        )

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )
        out = self.fc_out(out)
        dec_target = trg[:,1:].clone()
        dec_target[trg[:,1:]==PAD_ID]=-100
        loss = nn.CrossEntropyLoss()(out.view(-1, 10000)/1.0, dec_target.view(-1))
        
        
        return loss

