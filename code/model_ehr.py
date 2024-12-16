from math import sqrt
from tqdm import tqdm
import pdb
import math
import torch.nn.functional as F
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer, AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, QuantoConfig
import transformers
from torch.cuda.amp import autocast
transformers.logging.set_verbosity_error()

import torch.nn as nn

# 실험 설명: 시퀀스 길이 증가에 따른 연산량 지수적 증가를 예방하기 위해 프롬프트를 지워보기. 
# demographic 정보 통합 추가 코드는 modified로 라벨링 되어 있음

class TimeBatchNorm2d(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.norm = nn.BatchNorm2d(shape[1])  # shape: (sequence_length, channels)

    def forward(self, x):
        # x: (N, C, L)
        x = x.unsqueeze(2)  # (N, C, 1, L)
        x = self.norm(x)
        x = x.squeeze(2)  # (N, C, L)
        return x

class Feature_Mixing(nn.Module):
    def __init__(self, N, d_llm, d_ff, head_dropout=0, normalize_before: bool = True, norm_type: type = TimeBatchNorm2d):
        super().__init__()
        
        self.norm_before = (
            norm_type((d_llm, N))
            if normalize_before
            else nn.Identity()
        )
        self.norm_after = (
            norm_type((d_llm, N))
            if not normalize_before
            else nn.Identity()
        )
        
        self.fc1 = nn.Linear(N, d_ff)
        self.fc2 = nn.Linear(d_ff, N)
        self.activation_fn = nn.ReLU()
        self.dropout = nn.Dropout(head_dropout)
        
    def forward(self, x):
        # x -> B, N, d_llm
        x_proj = self.norm_before(x)
        x_proj = x_proj.permute(0, 2, 1)

        x = self.fc1(x_proj)  # (B, d_llm, d_ff)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)  # (B, d_llm, N)
        x = self.dropout(x)
        out = x_proj + x
        
        return out.permute(0, 2, 1)
    
class Dimension_Mixing(nn.Module):
    def __init__(self, N, d_llm, d_ff, head_dropout=0, normalize_before: bool = True, norm_type: type = TimeBatchNorm2d):
        super().__init__()
        
        self.norm_before = (
            norm_type((d_llm, N))
            if normalize_before
            else nn.Identity()
        )
        self.norm_after = (
            norm_type((d_llm, N))
            if not normalize_before
            else nn.Identity()
        )
        
        self.fc1 = nn.Linear(d_llm, d_ff)
        self.fc2 = nn.Linear(d_ff, d_ff)
        self.activation_fn = nn.ReLU()
        self.dropout = nn.Dropout(head_dropout)
        
    def forward(self, x):
        # x -> B, N, d_llm
        x_proj = self.norm_before(x)

        x = self.fc1(x_proj)  # (B, N, d_ff)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)  # (B, N, d_ff)
        out = self.dropout(x)
        return out
    
class Classifier(nn.Module):
    def __init__(self, N, d_llm, d_ff, target_window = 2, head_dropout=0.01, normalize_before: bool = True, norm_type: type = TimeBatchNorm2d):
        super().__init__()
        self.feature_mixing = Feature_Mixing(N, d_llm, d_ff, head_dropout, normalize_before, norm_type)
        self.dimension_mixing = Dimension_Mixing(N, d_llm, d_ff, head_dropout, normalize_before, norm_type)
        

        self.final_norm = nn.BatchNorm1d(N*d_ff + N)
        self.final_dropout = nn.Dropout(head_dropout)
        self.activation_fn = nn.ReLU()
        self.fc_final = nn.Linear(N * d_ff + N, target_window)
    
    def forward(self, x, emb):
        # Feature Mixing
        x = self.feature_mixing(x)  # (B, N, d_llm)
        
        # Dimension Mixing
        x = self.dimension_mixing(x)  # (B, N, d_ff)
        
        # classification
        x = x.view(x.size(0), -1)  # (B, N * d_ff)
        x = torch.cat([x, emb], dim=1) # (B, N * d_ff + N) modified
        x = self.final_norm(x) 
        x = self.activation_fn(x)
        x = self.final_dropout(x) 
        x = self.fc_final(x)  # (B, d_ff)
        
        return x

class Ehrtimellm(nn.Module):

    def __init__(self, configs):
        super(Ehrtimellm, self).__init__()
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.d_llm = configs.llm_dim
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.enc_in = configs.enc_in
        self.d_static = configs.d_static
        self.d_inp = configs.enc_in
        self.n_classes = configs.n_classes
        self.batch_size = configs.batch_size
        self.n_token = configs.num_tokens
        # self.quantization_config = QuantoConfig(weights="int8")
        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained("huggyllama/llama-7b")
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            
            try:
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    "huggyllama/llama-7b",
                    trust_remote_code=True,
                    config=self.llama_config,
                    ignore_mismatched_sizes=True,
                    local_files_only=True,
                    # load_in_4bit=True,
                    # quantization_config = self.quantization_config

                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    "huggyllama/llama-7b",
                    trust_remote_code=True,
                    config=self.llama_config,
                )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    "huggyllama/llama-7b",
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    "huggyllama/llama-7b",
                    trust_remote_code=True,
                    # local_files_only=False
                )
                
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    "openai-community/gpt2",
                    trust_remote_code=True,
                    # local_files_only=True,
                    config=self.gpt2_config,
                    # load_in_8bit=True
                    
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    "openai-community/gpt2",
                    trust_remote_code=True,
                    # local_files_only=False,
                    config=self.gpt2_config,
                    
                )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    "openai-community/gpt2",
                    trust_remote_code=True,
                    # local_files_only=True
                    # load_in_8bit=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    "openai-community/gpt2",
                    trust_remote_code=True,
                    # local_files_only=False
                )
                
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    # local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(configs.dropout)
        
        # static
        self.emb = nn.Linear(self.d_static, self.d_inp) # modified
        
        # Reprogramming
        self.reprogramming_layer = ReprogrammingLayer(self.d_ff, self.n_heads, d_keys = self.d_ff, d_llm=self.d_llm)
        
        # Text Prototype
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        
        # Variable aggragation
        self.clf = Classifier(self.enc_in, self.d_llm, self.d_ff)
    
    def forward(self, x_enc, time, real_time, demo, null_mask, mask=None):
        dec_out = self.forecast(x_enc, time, real_time, demo, null_mask)
        return dec_out #B, pred_window

    def forecast(self, x_enc, time, real_time, demo, null_mask):

        x_masked = x_enc * null_mask    
        
        # prompt as prefix
        B, T, N = x_masked.size()
        S = self.n_token
        D = self.d_llm
        
        emb = self.emb(demo) # B x N modified
       
        # Missing value control
        input_ids = self.tokenizer.encode('missing', return_tensors='pt')
        mask_token = self.llm_model.get_input_embeddings()(input_ids.to(x_enc.device))
        mask_token = mask_token[0].to(x_enc.dtype)

        # Text Prototype
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0).to(x_enc.dtype)).permute(1, 0)
        
        # Reprogramming
        rep_out = self.reprogramming_layer(x_masked, source_embeddings, source_embeddings)
        null_position = (rep_out.sum(dim=-1) == 0)
        rep_out[null_position] = mask_token.expand(null_position.sum(), -1)
        
        # Make attention mask for LLM
        nan_mask = null_mask == 0
        processed_tensor = torch.where(nan_mask, torch.tensor(0.0), null_mask)
        processed_tensor = torch.nan_to_num(processed_tensor, nan=1)
        attention_mask = processed_tensor.permute(0, 2, 1).reshape(B * N, T)
        
        rep_out = rep_out.to(x_enc.device)
        attention_mask = attention_mask.to(x_enc.device)
        
        # LLM Body
        chunk_size = self.batch_size
        total_size = rep_out.size(0) # batch size x enc_in
        repeated_real_time = real_time.unsqueeze(1).repeat(1, self.enc_in).flatten().to(rep_out.device)
        repeated_real_time = torch.clamp(repeated_real_time, 0, 59)
        outputs = []
        
        # import time
        # start_time = time.time()
        for i in range(0, total_size, chunk_size):
            end_idx = min(i + chunk_size, total_size)
            chunk_input = rep_out[i:end_idx]
            chunk_mask = attention_mask[i:end_idx]
            chunk_real_time = repeated_real_time[i:end_idx]
           
            with torch.no_grad():
                out = self.llm_model(inputs_embeds=chunk_input, attention_mask=chunk_mask).hidden_states[-1]  # (chunk_size, seq_len, d_llm
            last_outputs = out[:, -60:]  # (chunk_size, d_llm), take last sequence
            
            batch_indices = torch.arange(chunk_real_time.size(0), device=chunk_real_time.device)
            last_outputs = last_outputs[batch_indices, chunk_real_time]
            outputs.append(last_outputs)
        # elapsed_time = time.time() - start_time
        # print(f"Processing time: {elapsed_time:.2f} seconds") # w/o tokenlearner: batch size 16기준 6초 -> w tokenlearner: batch size 16기준  0.47
        
        final_output = torch.cat(outputs, dim=0) # B*S, d_llm

        del nan_mask, processed_tensor, attention_mask, outputs  
        
        # Classification
        latent = final_output.view(B, N, self.d_llm)
        dec_out = self.clf(latent, emb)
        
        return dec_out

class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.01):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        self.d_keys = d_keys
        self.n_heads = n_heads
        
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm) # mask 된 부분은 아예 무시 되도록 bias 계산 x
        
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, time_series, source_embedding, value_embedding):
        
        B, T, N = time_series.shape # B, T, N
        S, _ = source_embedding.shape
        H = self.n_heads
        
        # Query
        time_series = time_series.permute(0, 2, 1).reshape(B*N, T)
        repeated_ts = time_series.unsqueeze(-1).repeat(1, 1, self.d_keys * self.n_heads)
        
        nan_mask = torch.isnan(repeated_ts)
        Q = torch.where(nan_mask, torch.tensor(0.0), repeated_ts).view(B*N, T, H, -1) #B*N, T, H, d_ff
        
        # Key
        K = self.key_projection(source_embedding).view(S, H, -1)
        
        # Value
        V = self.value_projection(value_embedding).view(S, H, -1)

        out, mask = self.reprogramming(Q, K, V) # B_N, T, H, E

        out = out.reshape(B*N, T, -1)
        out = self.dropout(self.out_projection(out))

        result = out * mask.unsqueeze(-1)
        
        return result

    def reprogramming(self, Q, K, V):
        B_N, T, H, d_ff = Q.shape
        
        scale = 1. / sqrt(d_ff)
        scores = torch.einsum("bthe,she->bhts", Q, K)
        
        # Null Position masking 
        attention_mask = (scores != 0).float()
        temporal_masking = torch.sum(attention_mask, axis = -1)
        temporal_masking = (temporal_masking != 0).float() # B_N, H, T
        mask = temporal_masking.max(dim=1)[0] # B_N, T
        
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhts,she->bthe", A, V)
    
        return reprogramming_embedding, mask
    
    
# class Variable_Attention(nn.Module):
#     def __init__(self, D, n_heads, B, N, d_keys=None, attention_dropout=0.1):
#         super(Variable_Attention, self).__init__()

#         d_keys = d_keys or (D // n_heads)
#         self.d_keys = d_keys
#         self.B = B
#         self.N = N
#         self.query_projection = nn.Linear(D, d_keys * n_heads, bias=True)
#         self.key_projection = nn.Linear(D, d_keys * n_heads, bias=True)
#         self.value_projection = nn.Linear(D, d_keys * n_heads, bias=True)
#         self.summary_projection = nn.Linear(N, 1)
#         self.out_projection_layer = nn.Linear(D * n_heads, D, bias = True)
#         self.n_heads = n_heads
#         self.dropout = nn.Dropout(attention_dropout)

#     def forward(self, target_embedding):
#         B, N, D = target_embedding.shape # B, N, D

#         H = self.n_heads
#         Q = self.query_projection(target_embedding).view(B, N, H, self.d_keys) # B, N, H, E
#         K = self.key_projection(target_embedding).view(B, N, H, self.d_keys) # B, N, H, E
#         V = self.value_projection(target_embedding).view(B, N, H, self.d_keys) # B, N, H, E
  
#         beta = self.get_attention(Q, K) # B, H, N, 1
#         beta_ori = beta.squeeze(-1) # B, H, N
#         beta_cal = beta_ori.permute(0, 2, 1) # B, N, H
        
#         weighted_tensor = V * beta_cal.unsqueeze(-1)
#         result = weighted_tensor.sum(dim=1)

#         out = self.out_projection_layer(result.view(B, -1))
#         return self.dropout(out)
    
#     def get_attention(self, Q, K):
        
#         B, N, H, E = Q.shape
#         B, N, H, E = K.shape

#         scale = 1. / torch.sqrt(torch.tensor(E))

#         scores = torch.einsum("bnhd,bmhd->bhnm", Q, K) # B, H, N, N
#         beta = self.dropout(torch.softmax(self.summary_projection(scale * scores), dim=-2)) # B, H, N, 1
#         return beta

