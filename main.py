import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from dataset import YelpDataset
from model import MultiModalRecommender
from transformers import BertTokenizer, BertModel
from tqdm import tqdm


def load_data():
    # Load the datasets
    reviews_iter = pd.read_json('yelp_dataset/yelp_academic_dataset_review.json', lines=True, chunksize=1000)
    reviews_df = next(reviews_iter)
    users_iter = pd.read_json('yelp_dataset/yelp_academic_dataset_user.json', lines=True, chunksize=1000)
    users_df = next(users_iter)
    businesses_iter = pd.read_json('yelp_dataset/yelp_academic_dataset_business.json', lines=True, chunksize=1000)
    businesses_df = next(businesses_iter)
    

    # Select relevant columns
    reviews_df = reviews_df[['review_id', 'user_id', 'business_id', 'stars', 'text']]
    users_df = users_df[['user_id', 'average_stars', 'review_count', 'useful', 'funny', 'cool']]
    businesses_df = businesses_df[['business_id', 'stars', 'review_count', 'is_open']]
    
    return reviews_df, users_df, businesses_df


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            user_features = batch['user_features'].to(device)
            business_features = batch['business_features'].to(device)
            ratings = batch['rating'].to(device)
            
            optimizer.zero_grad()
            predictions = model(input_ids, attention_mask, user_features, business_features)
            loss = criterion(predictions, ratings)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                user_features = batch['user_features'].to(device)
                business_features = batch['business_features'].to(device)
                ratings = batch['rating'].to(device)
                
                predictions = model(input_ids, attention_mask, user_features, business_features)
                loss = criterion(predictions, ratings)
                val_loss += loss.item()
        
        print(f'Epoch {epoch+1}:')
        print(f'Training Loss: {train_loss/len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    print("Loading data...")
    reviews_df, users_df, businesses_df = load_data()
    valid_user_ids = set(users_df['user_id'])
    valid_business_ids = set(businesses_df['business_id'])

    reviews_df = reviews_df[
        reviews_df['user_id'].isin(valid_user_ids) & 
        reviews_df['business_id'].isin(valid_business_ids)
    ]
    # Split data
    train_reviews, val_reviews = train_test_split(reviews_df, test_size=0.2, random_state=42)
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets
    train_dataset = YelpDataset(train_reviews, users_df, businesses_df, tokenizer)
    val_dataset = YelpDataset(val_reviews, users_df, businesses_df, tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize model
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    model = MultiModalRecommender(bert_model).to(device)
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    
    # Train model
    print("Starting training...")
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5, device=device)


if __name__ == "__main__":
    main() 