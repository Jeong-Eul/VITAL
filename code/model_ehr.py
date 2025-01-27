from math import sqrt
from tqdm import tqdm
import pdb
import math
import torch.nn.functional as F
from torch.cuda.amp import autocast
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

# 실험 설명: Left padding 하기
# demographic 정보 통합 추가 코드는 modified로 라벨링 되어 있음

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

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
    def __init__(self, N, d_ff, head_dropout=0.1, normalize_before: bool = True, norm_type: type = TimeBatchNorm2d):
        super().__init__()
        
        self.norm_before = (
            norm_type((d_ff, N))
            if normalize_before
            else nn.Identity()
        )
        self.norm_after = (
            norm_type((d_ff, N))
            if not normalize_before
            else nn.Identity()
        )
        
        self.fc1 = nn.Linear(N, d_ff)
        self.fc2 = nn.Linear(d_ff, N)
        self.activation_fn = nn.ReLU()
        self.dropout = nn.Dropout(head_dropout)
        
    def forward(self, x):
        # x -> B, N, d_ff
        x_proj = self.norm_before(x)
        x_proj = x_proj.permute(0, 2, 1)

        x = self.fc1(x_proj)  # (B, d_ff, d_ff)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)  # (B, d_ff, N)
        x = self.dropout(x)
        out = x_proj + x
        
        return out.permute(0, 2, 1)
    
class Dimension_Mixing(nn.Module):
    def __init__(self, N, d_llm, d_ff, head_dropout=0.1, normalize_before: bool = True, norm_type: type = TimeBatchNorm2d):
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
    
class Classifier_v2(nn.Module):
    def __init__(self, N, N_vital, d_llm, d_ff, target_window = 2, head_dropout=0.1, normalize_before: bool = True, norm_type: type = TimeBatchNorm2d):
        super().__init__()
        self.feature_mixing = Feature_Mixing(N, d_ff, head_dropout, normalize_before, norm_type)
        self.dimension_mixing = Dimension_Mixing(N_vital, d_llm, d_ff, head_dropout, normalize_before, norm_type)
        

        self.final_norm = nn.LayerNorm(N * d_ff)
        self.final_dropout = nn.Dropout(head_dropout)
        self.activation_fn = nn.GELU()
        self.fc_1 = nn.Linear(N * d_ff, d_ff)
        self.fc_2 = nn.Linear(d_ff, target_window)
        
    def forward(self, x_vital, x_lab):

        # Dimension Mixing
        vital_emb = self.dimension_mixing(x_vital)  # (B, N_vital, d_ff)
        total_emb = torch.cat([vital_emb, x_lab], axis = 1)
        
        # Feature Mixing
        x_emb = self.feature_mixing(total_emb) # (B, N, d_ff)
        
        # classification
        x = x_emb.view(x_emb.size(0), -1)  # (B, N * d_ff)
        
        x = self.final_norm(x)
        x = self.activation_fn(x)
        x = self.final_dropout(x)

        x = self.fc_1(x)
        x = self.activation_fn(x)
        x = self.final_dropout(x)

        x = self.fc_2(x)
       
        return x


class Classifier(nn.Module):
    def __init__(self, N, N_vital, d_llm, d_ff, target_window = 2, head_dropout=0.1, normalize_before: bool = True, norm_type: type = TimeBatchNorm2d):
        super().__init__()
        self.feature_mixing = Feature_Mixing(N, d_ff, head_dropout, normalize_before, norm_type)
        self.dimension_mixing = Dimension_Mixing(N_vital, d_llm, d_ff, head_dropout, normalize_before, norm_type)
        

        self.final_norm = nn.LayerNorm(N * d_ff + d_ff)
        self.final_dropout = nn.Dropout(head_dropout)
        self.activation_fn = nn.GELU()
        self.fc_1 = nn.Linear(N * d_ff + d_ff, d_ff)
        self.fc_2 = nn.Linear(d_ff, target_window)
        
    def forward(self, x_vital, x_lab, emb):

        # Dimension Mixing
        vital_emb = self.dimension_mixing(x_vital)  # (B, N_vital, d_ff)
        total_emb = torch.cat([vital_emb, x_lab], axis = 1)
        
        # Feature Mixing
        x_emb = self.feature_mixing(total_emb) # (B, N, d_ff)
        
        # classification
        x = x_emb.view(x_emb.size(0), -1)  # (B, N * d_ff)
        x = torch.cat([x, emb], dim=1) # (B, N * d_ff + d_ff) modified
        
        x = self.final_norm(x)
        x = self.activation_fn(x)
        x = self.final_dropout(x)

        x = self.fc_1(x)
        x = self.activation_fn(x)
        x = self.final_dropout(x)

        x = self.fc_2(x)
       
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
        self.static = configs.static_info
        # self.quantization_config = QuantoConfig(weights="int8")
        if configs.llm_model == 'LLAMA3':
            self.llama_config = LlamaConfig.from_pretrained("meta-llama/Llama-3.1-8B")
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            
            try:
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    "meta-llama/Llama-3.1-8B",
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
                    "meta-llama/Llama-3.1-8B",
                    trust_remote_code=True,
                    config=self.llama_config,
                    # load_in_4bit=True,
                )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    "meta-llama/Llama-3.1-8B",
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    "meta-llama/Llama-3.1-8B",
                    trust_remote_code=True,
                    # local_files_only=False
                )
        
        elif configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained("huggyllama/llama-7b")
            # self.llama_config.num_hidden_layers = configs.llm_layers
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
        
        # variable information
        self.vital_index = configs.vital
        self.lab_index = configs.lab

        self.dropout = nn.Dropout(configs.dropout)
        
        # static
        if self.static:
            self.emb = nn.Linear(self.d_static, self.d_ff) 
            
        self.lab_embedder = LabStat_Embedder(len(self.lab_index), self.d_ff)
        self.nm_token = nn.Parameter(torch.empty(1, self.d_ff))
        nn.init.xavier_uniform_(self.nm_token)
        
        # Reprogramming
        self.reprogramming_layer = ReprogrammingLayer(self.d_ff, self.n_heads, d_keys = self.d_ff, d_llm=self.d_llm)
        
        # Text Prototype
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        
        # Classification
        if self.static:
            self.clf = Classifier(self.enc_in, len(self.vital_index), self.d_llm, self.d_ff)
        else:
            self.clf = Classifier_v2(self.enc_in, len(self.vital_index), self.d_llm, self.d_ff, target_window = self.n_classes)
    
    def forward(self, x_enc, time, real_time, demo, null_mask, mask=None):
        dec_out = self.forecast(x_enc, time, real_time, demo, null_mask)
        return dec_out #B, pred_window

    def forecast(self, x_enc, time, real_time, demo, null_mask):

        x_masked = x_enc * null_mask   
        
        x_vital = x_masked[:, :, self.vital_index]
        N_vital = x_vital.size()[-1]

        x_lab = x_masked[:, :, self.lab_index]
        N_lab = x_lab.size()[-1]
        
        # prompt as prefix
        B, T, N = x_masked.size()
        D = self.d_llm
        
        if self.static:
            emb = self.emb(demo) # B x d_ff modified
       
        # Missing value control
        input_ids = self.tokenizer.encode('missing', return_tensors='pt')
        mask_token = self.llm_model.get_input_embeddings()(input_ids.to(x_masked.device))
        mask_token = mask_token[0].to(x_masked.dtype)
        # mask_token = mask_token[0][1].to(x_masked.device)
        
        # Padding token control 
        input_ids = self.tokenizer.encode(self.tokenizer.pad_token, return_tensors='pt')
        pad_token = self.llm_model.get_input_embeddings()(input_ids.to(x_masked.device))
        # pad_token = pad_token[0][1].to(x_masked.device)
        pad_token = pad_token[0].to(x_masked.device)

        # Text Prototype
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0).to(x_masked.device)).permute(1, 0)
    
        # Reprogramming
        rep_out = self.reprogramming_layer(x_vital, source_embeddings, source_embeddings)
    
        # sequencing
        result_tensor = torch.zeros(B*N_vital,  T, self.d_llm).to(x_masked.device).to(x_masked.dtype)
        attention_mask = torch.zeros(B*N_vital, T, dtype=torch.float32).to(x_masked.device).to(x_masked.dtype)
            
        for i in range(B):
            start_idx = i * N_vital
            end_idx = (i + 1) * N_vital
            patient_data = rep_out[start_idx:end_idx]
            actual_length = real_time[i]
            
            reshaped_data = patient_data[:, :actual_length, :] # [NumVars, real_time, LLM Dim]
            
            padding_length = T - reshaped_data.size(1)
            padding_token = pad_token.repeat(reshaped_data.size(0), padding_length, 1) # [NumVars, T - real_time, LLM Dim]

            padded_data = torch.cat((padding_token, reshaped_data), dim=1)  # [NumVars,  SeqLen, LLM Dim]

            result_tensor[start_idx:end_idx] = padded_data
            attention_mask[start_idx:end_idx, padding_length:] = 1  
        
        null_position = (result_tensor.sum(dim=-1) == 0)
        result_tensor[null_position] = mask_token.expand(null_position.sum(), -1)
        
        # LLM Body
        chunk_size = 8
        total_size = B*N_vital
        outputs = []
        
        for i in range(0, total_size, chunk_size):
            chunk_input = result_tensor[i : i + chunk_size]
            chunk_mask = attention_mask[i : i + chunk_size]
            
            with torch.no_grad():
                out = self.llm_model(inputs_embeds=chunk_input).hidden_states[-1]  # (chunk_size, seq_len, d_llm)
            last_outputs = out[:, -1]  # (chunk_size, d_llm), take last sequence
            outputs.append(last_outputs)
            
        final_output = torch.cat(outputs, dim=0) # B*N_vital, d_llm
        
        # Lab statistic embedding
        lab_stats = compute_statistics(x_lab)
        not_measured_mask = (lab_stats == 0).all(dim=1)
        
        lab_emb = self.lab_embedder(lab_stats)
        lab_emb[not_measured_mask] = self.nm_token # B, N_lab, d_ff
        
        # Classification
        latent = final_output.view(B, N_vital, self.d_llm)
        
        if self.static:
            dec_out = self.clf(latent, lab_emb, emb)
        else:
            dec_out = self.clf(latent, lab_emb)
        
        return dec_out

class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.01):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        self.d_keys = d_keys
        self.n_heads = n_heads
        
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        
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
    

def compute_statistics(tensor):
    """
    Compute mean, median, min, max, and std for each variable in the tensor.

    Args:
        tensor (torch.Tensor): Input tensor with shape (B, N, T), where:
                               - B: Batch size
                               - N: Number of variables
                               - T: Sequence length
                               - NaN indicates missing values
                               - Padding is indicated by 0

    Returns:
        torch.Tensor: Tensor of shape (B, 5, N) with the computed statistics.
                      - 5 corresponds to [mean, median, min, max].
    """
    B, T, N = tensor.shape

    # Create masks for non-NaN and non-padding values
    valid_mask = (tensor != 0) & (~torch.isnan(tensor))  # True for valid values

    # Replace invalid values (NaN and padding) with a large negative value for min/max and zero for others
    processed_tensor = tensor.clone()
    processed_tensor[~valid_mask] = float('-inf')

    # Compute statistics
    mean = torch.nanmean(processed_tensor.masked_fill(~valid_mask, float('nan')), dim=1)  # Mean along time dimension
    median = torch.nanmedian(processed_tensor.masked_fill(~valid_mask, float('nan')), dim=1).values  # Median along time dimension
    min_val = torch.min(processed_tensor.masked_fill(~valid_mask, float('inf')), dim=1).values  # Min along time dimension
    max_val = torch.max(processed_tensor.masked_fill(~valid_mask, float('-inf')), dim=1).values  # Max along time dimension

    # Replace invalid statistics (if all values are invalid) with 0
    mean[~valid_mask.any(dim=1)] = 0
    median[~valid_mask.any(dim=1)] = 0
    min_val[~valid_mask.any(dim=1)] = 0
    max_val[~valid_mask.any(dim=1)] = 0

    # Stack results into (B, 5, N)
    stats = torch.stack([mean, median, min_val, max_val], dim=1)

    return stats

class LabStat_Embedder(nn.Module):
    def __init__(self, N, d_ff, head_dropout = 0.1, S = 4):
        super().__init__()
        
        self.fc1 = nn.Linear(S, d_ff)
        self.fc2 = nn.Linear(d_ff, d_ff)
        self.activation_fn = nn.ReLU()
        self.dropout = nn.Dropout(head_dropout)
        
    def forward(self, x):
        # x -> B, S, N
        x = x.permute(0, 2, 1) # B, N, S
        x = self.fc1(x)  # (B, N, d_ff)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)  # (B, d_ff, d_ff)
        return x