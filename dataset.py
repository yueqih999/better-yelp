import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer
from textblob import TextBlob


class YelpDataset(Dataset):
    def __init__(self, reviews_df, users_df, businesses_df, user_texts, business_texts, tokenizer, max_length=256):
        self.reviews = reviews_df.reset_index(drop=True)
        self.users = users_df
        self.businesses = businesses_df
        self.user_texts = user_texts       
        self.business_texts = business_texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Precompute sentiment scores for all user texts
        self.user_sentiments = {}
        for user_id, text in user_texts.items():
            sentiment = self._compute_sentiment(text)
            self.user_sentiments[user_id] = sentiment
        
    def _compute_sentiment(self, text):
        """Compute sentiment scores using TextBlob"""
        if not text:
            return torch.tensor([0.0, 0.0], dtype=torch.float)
        
        blob = TextBlob(text)
        # Get polarity (-1 to 1) and subjectivity (0 to 1)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        return torch.tensor([polarity, subjectivity], dtype=torch.float)
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review = self.reviews.iloc[idx]
        user_id = review['user_id']
        business_id = review['business_id']

        # Process user text
        user_text = self.user_texts.get(user_id, "")
        user_enc = self.tokenizer(
            user_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        user_input_ids = user_enc['input_ids'].squeeze(0)
        user_attention_mask = user_enc['attention_mask'].squeeze(0)

        # Get user sentiment
        user_sentiment = self.user_sentiments.get(
            user_id, 
            torch.tensor([0.0, 0.0], dtype=torch.float)
        )

        # Process business text
        business_text = self.business_texts.get(business_id, "")
        business_enc = self.tokenizer(
            business_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        business_input_ids = business_enc['input_ids'].squeeze(0)
        business_attention_mask = business_enc['attention_mask'].squeeze(0)

        # Get user features
        user_row = self.users[self.users['user_id'] == user_id]
        if user_row.empty:
            user_info = None
        else:
            user_info = user_row.iloc[0]

        # Get business features
        business_row = self.businesses[self.businesses['business_id'] == business_id]
        if business_row.empty:
            business_info = None
        else:
            business_info = business_row.iloc[0]

        if user_info is None or business_info is None:
            return self.__getitem__((idx + 1) % len(self.reviews))

        # Create user features (5 features + 2 sentiment features)
        user_features = torch.tensor([
            user_info['average_stars'],
            user_info['review_count'],
            user_info['useful'],
            user_info['funny'],
            user_info['cool']
        ], dtype=torch.float)

        # Reshape user features to (1, 5) for concatenation
        user_features = user_features.unsqueeze(0)
        
        # Reshape sentiment to (1, 2) for concatenation
        user_sentiment = user_sentiment.unsqueeze(0)
        
        # Concatenate along the feature dimension
        user_features = torch.cat([user_features, user_sentiment], dim=1)
        
        # Remove the extra dimension
        user_features = user_features.squeeze(0)

        # Create business features
        business_features = torch.tensor([
            business_info['stars'],
            business_info['review_count'],
            business_info['is_open']
        ], dtype=torch.float)

        # Verify dimensions
        if user_features.shape[0] != 7:
            raise ValueError(f"Expected user features dimension 7, got {user_features.shape[0]}")
        if business_features.shape[0] != 3:
            raise ValueError(f"Expected business features dimension 3, got {business_features.shape[0]}")

        return {
            "user_input_ids": user_input_ids,
            "user_attention_mask": user_attention_mask,
            "user_features": user_features,

            "business_input_ids": business_input_ids,
            "business_attention_mask": business_attention_mask,
            "business_features": business_features,
        }



if __name__ == "__main__":
    reviews_iter = pd.read_json('yelp_dataset/yelp_academic_dataset_review.json', lines=True, chunksize=100)
    reviews_df = next(reviews_iter)
    users_iter = pd.read_json('yelp_dataset/yelp_academic_dataset_user.json', lines=True, chunksize=100)
    users_df = next(users_iter)
    businesses_iter = pd.read_json('yelp_dataset/yelp_academic_dataset_business.json', lines=True, chunksize=100)
    businesses_df = next(businesses_iter)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = YelpDataset(reviews_df, users_df, businesses_df, tokenizer)
    print(len(dataset))
