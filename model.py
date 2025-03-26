import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import json
from tqdm import tqdm


class MultiModalTwoTower(nn.Module):
    def __init__(self, bert_model, user_feature_dim=5, business_feature_dim=3, hidden_dim=128):
        super().__init__()
        self.bert = bert_model
        self.text_dim = 768  # BERT hidden size
        
        # Feature processing layers
        self.user_mlp = nn.Sequential(
            nn.Linear(user_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim)
        )
        
        self.business_mlp = nn.Sequential(
            nn.Linear(business_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim)
        )
        
        self.bert_proj = nn.Linear(self.text_dim, hidden_dim)
        

    def encode_user(self, user_features, user_input_ids, user_attention_mask):
        text_embed = self.bert(user_input_ids, user_attention_mask).last_hidden_state[:, 0, :]
        text_embed = self.bert_proj(text_embed)
        struct_embed = self.user_mlp(user_features)
        return nn.functional.normalize(text_embed + struct_embed, dim=1)


    def encode_business(self, business_features, business_input_ids, business_attention_mask):
        text_embed = self.bert(business_input_ids, business_attention_mask).last_hidden_state[:, 0, :]
        text_embed = self.bert_proj(text_embed)
        struct_embed = self.business_mlp(business_features)
        return nn.functional.normalize(text_embed + struct_embed, dim=1)


    def forward(self, 
                user_features, 
                user_input_ids,
                user_attention_mask, 
                business_features,
                business_input_ids,
                business_attention_mask):
        user_emb = self.encode_user(user_features, user_input_ids, user_attention_mask)
        business_emb = self.encode_business(business_features, business_input_ids, business_attention_mask)
        return user_emb, business_emb
