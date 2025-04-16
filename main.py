import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from dataset import YelpDataset
from model import MultiModalTwoTower
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from collections import defaultdict
import os
from retrieval import FaissRetriever


def load_data():
    reviews_df = pd.read_json('yelp_dataset/yelp_academic_dataset_review.json', lines=True, nrows=10000)
    users_df = pd.read_json('yelp_dataset/yelp_academic_dataset_user.json', lines=True, nrows=10000)
    businesses_df = pd.read_json('yelp_dataset/yelp_academic_dataset_business.json', lines=True, nrows=10000)

    reviews_df = reviews_df[['review_id', 'user_id', 'business_id', 'stars', 'text']]
    users_df = users_df[['user_id', 'average_stars', 'review_count', 'useful', 'funny', 'cool']]
    businesses_df = businesses_df[['business_id', 'stars', 'review_count', 'is_open']]
    
    return reviews_df, users_df, businesses_df

def aggregate_user_texts(reviews_df, max_reviews_per_user=50):
    user_texts = defaultdict(list)

    for _, row in reviews_df.iterrows():
        user_id = row['user_id']
        text = row['text']

        if len(user_texts[user_id]) < max_reviews_per_user:
            user_texts[user_id].append(text)

    for uid in user_texts:
        user_texts[uid] = " ".join(user_texts[uid]) 

    return dict(user_texts)


def collect_business_categories(businesses_df):
    business_texts = {}
    for _, row in businesses_df.iterrows():
        b_id = row['business_id']
        cats = row.get('categories', "")
        if not isinstance(cats, str):
            cats = "" 
        business_texts[b_id] = cats
    return business_texts


def train_model(model, train_loader, val_loader, optimizer, num_epochs, device):
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            user_input_ids = batch['user_input_ids'].to(device)
            user_attention_mask = batch['user_attention_mask'].to(device)
            user_features = batch['user_features'].to(device)

            business_input_ids = batch['business_input_ids'].to(device)
            business_attention_mask = batch['business_attention_mask'].to(device)
            business_features = batch['business_features'].to(device)
            
            optimizer.zero_grad()
            user_emb, business_embed = model(
                user_features, user_input_ids, user_attention_mask,
                business_features, business_input_ids, business_attention_mask
            )

            logits = torch.matmul(user_emb, business_embed.T)
            labels = torch.arange(logits.size(0)).long().to(device)  # [0,1,2,... B-1]
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                user_input_ids = batch['user_input_ids'].to(device)
                user_attention_mask = batch['user_attention_mask'].to(device)
                user_features = batch['user_features'].to(device)

                business_input_ids = batch['business_input_ids'].to(device)
                business_attention_mask = batch['business_attention_mask'].to(device)
                business_features = batch['business_features'].to(device)
                
                user_emb, business_embed = model(
                    user_features, user_input_ids, user_attention_mask,
                    business_features, business_input_ids, business_attention_mask
                )

                logits = torch.matmul(user_emb, business_embed.T)
                labels = torch.arange(logits.size(0)).long().to(device)  # [0,1,2,... B-1]
            
                loss = criterion(logits, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)

        print(f'Epoch {epoch+1}:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'model/best_model.pth')
            print("  - Best model saved.")


def encode_all_businesses(model, businesses_df, business_texts, tokenizer, device, batch_size=32):
    """Encode all businesses using the trained model"""
    model.eval()
    all_embeddings = []
    all_business_ids = []
    business_info = {}
    

    for i in range(0, len(businesses_df), batch_size):
        batch_df = businesses_df.iloc[i:i+batch_size]
        batch_ids = batch_df['business_id'].tolist()

        batch_texts = [business_texts.get(bid, "") for bid in batch_ids]
        
        # Tokenize
        encodings = tokenizer(
            batch_texts,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(device)
        
        # Create business features
        features = torch.tensor([
            [row['stars'], row['review_count'], row['is_open']]
            for _, row in batch_df.iterrows()
        ], dtype=torch.float).to(device)
        
        # Get embeddings
        with torch.no_grad():
            _, business_embed = model(
                None, 
                None, 
                None, 
                features,
                encodings['input_ids'],
                encodings['attention_mask']
            )
            all_embeddings.append(business_embed.cpu())
            all_business_ids.extend(batch_ids)
        
        # Store business info
        for _, row in batch_df.iterrows():
            business_info[row['business_id']] = {
                'name': row.get('name', ''),
                'stars': row['stars'],
                'review_count': row['review_count'],
                'categories': business_texts.get(row['business_id'], '')
            }
    
    return torch.cat(all_embeddings), all_business_ids, business_info

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading data...")
    reviews_df, users_df, businesses_df = load_data()
    valid_user_ids = set(users_df['user_id'])
    valid_business_ids = set(businesses_df['business_id'])
    reviews_df = reviews_df[
        reviews_df['user_id'].isin(valid_user_ids) & 
        reviews_df['business_id'].isin(valid_business_ids)
    ]

    user_texts = aggregate_user_texts(reviews_df, max_reviews_per_user=50)
    business_texts = collect_business_categories(businesses_df)

    train_reviews, val_reviews = train_test_split(reviews_df, test_size=0.2, random_state=42)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = YelpDataset(train_reviews, users_df, businesses_df, user_texts, business_texts, tokenizer)
    val_dataset = YelpDataset(val_reviews, users_df, businesses_df, user_texts, business_texts, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    bert_model = BertModel.from_pretrained('bert-base-uncased')
    model = MultiModalTwoTower(
        bert_model,
        user_feature_dim=7,
        business_feature_dim=3,
        hidden_dim=128
    ).to(device)

    best_model_path = 'model/best_model.pth'
    if os.path.exists(best_model_path):
        print(f"Loading existing model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
        print("Model loaded successfully")
    else:
        print("No existing model found. Starting training from scratch.")

    print("reviews_df after filter:", len(reviews_df))
    print("train_reviews:", len(train_reviews))
    print("val_reviews:", len(val_reviews))
    print("train_dataset:", len(train_dataset))
    print("val_dataset:", len(val_dataset))

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    print("Starting training...")
    train_model(model, train_loader, val_loader, optimizer=optimizer, num_epochs=15, device=device)

    # build the retrieval index
    print("\nBuilding retrieval index...")
    retriever = FaissRetriever(hidden_dim=128)
    
    print("Encoding all businesses...")
    business_embeddings, business_ids, business_info = encode_all_businesses(
        model, businesses_df, business_texts, tokenizer, device
    )
    # Add to index
    print("Adding businesses to index...")
    retriever.add_businesses(business_embeddings, business_ids, business_info)
    
    # Example
    print("\nTesting recommendations...")
    sample_user = val_dataset[0]
    user_features = sample_user['user_features'].unsqueeze(0).to(device)
    user_input_ids = sample_user['user_input_ids'].unsqueeze(0).to(device)
    user_attention_mask = sample_user['user_attention_mask'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        user_emb, _ = model(
            user_features,
            user_input_ids,
            user_attention_mask,
            None,  # business features not needed
            None,  # business input ids not needed
            None   # business attention mask not needed
        )
    
    user_emb = user_emb[0].cpu().detach()
    
    # Get recommendations
    recommendations = retriever.search(user_emb)
    print("\nTop 5 recommendations:")
    for business_id, score, info in recommendations[:5]:
        print(f"Business: {info.get('name', 'Unknown')}")
        print(f"Categories: {info.get('categories', 'N/A')}")
        print(f"Rating: {info.get('stars', 0.0):.1f} ({info.get('review_count', 0)} reviews)")
        print(f"Similarity Score: {score:.3f}")
        print()


if __name__ == "__main__":
    main() 