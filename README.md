# Yelp Business Recommendation System

A multimodal recommendation system that combines user and business information to provide personalized business recommendations using a two-tower neural network architecture with efficient Faiss-based retrieval.

## Dataset Source

The project uses the Yelp Academic Dataset, which includes:
- `yelp_academic_dataset_review.json`: Contains review text and ratings
- `yelp_academic_dataset_user.json`: Contains user information and statistics
- `yelp_academic_dataset_business.json`: Contains business information and categories

The dataset is processed to include:
- User reviews (up to 50 reviews per user)
- Business categories
- User features (average stars, review count, useful/funny/cool votes)
- Business features (stars, review count, open status)

## Model Architecture

### Two-Tower Network
1. User Tower:
   - BERT encoder for processing user review text
   - MLP layers for processing numerical user features (7-dim)
   - Feature fusion layer combining text and numerical embeddings

2. Business Tower:
   - BERT encoder for processing business categories
   - MLP layers for processing numerical business features (3-dim)
   - Feature fusion layer combining text and numerical embeddings

3. Shared Components:
   - Hidden dimension: 128
   - BERT base model (uncased)
   - L2 normalization on final embeddings
   - Cross-entropy loss for training

### Faiss Retrieval System
- IndexFlatIP index type for exact inner product similarity computation
- L2 normalized embeddings for cosine similarity search
- Supports both single query and batch retrieval

## Features

### Input Features
1. User Features:
   - Numerical (5-dim): average stars, review count, useful/funny/cool votes
   - Sentiment (2-dim): polarity and subjectivity scores from TextBlob
   - Text: Aggregated historical reviews (max 50 reviews)

2. Business Features:
   - Numerical (3-dim): stars, review count, open status
   - Text: Business categories

### Output
- User and business embeddings in a shared 128-dim space
- Fast similarity search using Faiss
- Ranked list of recommended businesses with metadata
- Similarity scores between users and businesses

## Training

The model is trained using:
- Batch size: 32
- Learning rate: 2e-5
- Optimizer: Adam
- Loss function: Cross-entropy
- Training/validation split: 80/20

## Inference

1. User Encoding:
   - Processes user features and review text
   - Generates 128-dim user embedding

2. Business Indexing:
   - Pre-computes all business embeddings
   - Builds Faiss index for fast retrieval

3. Recommendation:
   - Performs efficient similarity search
   - Returns top-k most similar businesses
   - Includes business metadata in results

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train model and build index:
```bash
python main.py
```

3. Get recommendations:
```python
# Single user recommendation
recommendations = retriever.search(user_embedding, k=10)

# Batch recommendations
batch_recommendations = retriever.batch_search(user_embeddings, k=10)
```