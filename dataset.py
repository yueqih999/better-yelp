import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer


class YelpDataset(Dataset):
    def __init__(self, reviews_df, users_df, businesses_df, tokenizer, max_length=128):
        self.reviews = reviews_df
        self.users = users_df
        self.businesses = businesses_df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review = self.reviews.iloc[idx]
        user = self.users[self.users['user_id'] == review['user_id']].iloc[0]
        business = self.businesses[self.businesses['business_id'] == review['business_id']].iloc[0]
        
        # Process review text
        text_encoding = self.tokenizer(
            review['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # User features
        user_features = torch.tensor([
            user['average_stars'],
            user['review_count'],
            user['useful'],
            user['funny'],
            user['cool']
        ], dtype=torch.float)
        
        # Business features
        business_features = torch.tensor([
            business['stars'],
            business['review_count'],
            business['is_open']
        ], dtype=torch.float)
        
        # Target rating
        rating = torch.tensor(review['stars'], dtype=torch.float)
        
        return {
            'input_ids': text_encoding['input_ids'].squeeze(),
            'attention_mask': text_encoding['attention_mask'].squeeze(),
            'user_features': user_features,
            'business_features': business_features,
            'rating': rating
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
