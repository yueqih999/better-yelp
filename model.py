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


class MultiModalRecommender(nn.Module):
    def __init__(self, bert_model, user_feature_dim=5, business_feature_dim=3):
        super().__init__()
        self.bert = bert_model
        self.text_dim = 768  # BERT hidden size
        
        # Feature processing layers
        self.user_mlp = nn.Sequential(
            nn.Linear(user_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        self.business_mlp = nn.Sequential(
            nn.Linear(business_feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # Combined features processing
        self.combined_mlp = nn.Sequential(
            nn.Linear(self.text_dim + 32 + 16, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        
    def forward(self, input_ids, attention_mask, user_features, business_features):
        # Process text through BERT
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_output.last_hidden_state[:, 0, :]  # Use [CLS] token
        
        # Process user and business features
        user_processed = self.user_mlp(user_features)
        business_processed = self.business_mlp(business_features)
        
        # Combine all features
        combined_features = torch.cat([text_features, user_processed, business_processed], dim=1)
        
        # Predict rating
        rating_pred = self.combined_mlp(combined_features)
        return rating_pred.squeeze()
