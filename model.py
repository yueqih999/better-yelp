import torch
import torch.nn as nn
from transformers import BertModel


class MultiModalTwoTower(nn.Module):
    def __init__(self, bert_model, user_feature_dim=7, business_feature_dim=3, hidden_dim=128):
        super().__init__()
        self.bert = bert_model
        self.text_dim = 768  # BERT hidden size
        
        # Store dimensions
        self.user_feature_dim = user_feature_dim
        self.business_feature_dim = business_feature_dim
        self.hidden_dim = hidden_dim
        
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

    def load_state_dict(self, state_dict, strict=True):
        """Override load_state_dict to handle dimension mismatch"""
        # Get the current user MLP weights
        current_user_weights = self.user_mlp[0].weight.data
        
        # Load the state dict
        super().load_state_dict(state_dict, strict=False)
        
        # If the loaded weights have different dimensions
        if current_user_weights.shape[1] != self.user_feature_dim:
            print("Adjusting model weights for new feature dimensions...")
            # Initialize new weights for the additional features
            new_weights = torch.zeros((64, self.user_feature_dim))
            new_weights[:, :current_user_weights.shape[1]] = current_user_weights
            self.user_mlp[0].weight.data = new_weights
            
            # Initialize new biases
            new_biases = torch.zeros(64)
            new_biases[:] = self.user_mlp[0].bias.data
            self.user_mlp[0].bias.data = new_biases

    def encode_user(self, user_features, user_input_ids, user_attention_mask):
        # Check dimensions
        if user_features.shape[1] != self.user_feature_dim:
            raise ValueError(
                f"Expected user features dimension {self.user_feature_dim}, "
                f"got {user_features.shape[1]}"
            )
            
        text_embed = self.bert(
            user_input_ids, 
            user_attention_mask
        ).last_hidden_state[:, 0, :]
        text_embed = self.bert_proj(text_embed)
        struct_embed = self.user_mlp(user_features)
        return nn.functional.normalize(text_embed + struct_embed, dim=1)


    def encode_business(self, business_features, business_input_ids, business_attention_mask):
        # Check dimensions
        if business_features.shape[1] != self.business_feature_dim:
            raise ValueError(
                f"Expected business features dimension {self.business_feature_dim}, "
                f"got {business_features.shape[1]}"
            )
            
        text_embed = self.bert(
            business_input_ids, 
            business_attention_mask
        ).last_hidden_state[:, 0, :]
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
        user_emb = self.encode_user(
            user_features, 
            user_input_ids, 
            user_attention_mask
        )
        business_emb = self.encode_business(
            business_features, 
            business_input_ids, 
            business_attention_mask
        )
        return user_emb, business_emb
