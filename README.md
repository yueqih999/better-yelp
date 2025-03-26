# Yelp Business Recommendation System

A multimodal recommendation system that combines user and business information to provide personalized business recommendations using a two-tower neural network architecture.

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

## Input/Output

### Input
1. User Features:
   - Average stars
   - Review count
   - Useful votes
   - Funny votes
   - Cool votes
   - User review text (aggregated from past reviews)

2. Business Features:
   - Average stars
   - Review count
   - Open status
   - Business categories

### Output
- User and business embeddings in a shared latent space
- Similarity scores between users and businesses
- Business recommendations based on user preferences

## Model Structure

The model uses a two-tower architecture with the following components:

1. User Tower:
   - BERT encoder for processing user review text
   - MLP layers for processing numerical user features
   - Feature fusion layer combining text and numerical embeddings

2. Business Tower:
   - BERT encoder for processing business categories
   - MLP layers for processing numerical business features
   - Feature fusion layer combining text and numerical embeddings

3. Shared Components:
   - Hidden dimension: 128
   - BERT base model (uncased)
   - L2 normalization on final embeddings
   - Cross-entropy loss for training

## Training

The model is trained using:
- Batch size: 32
- Learning rate: 2e-5
- Optimizer: Adam
- Loss function: Cross-entropy
- Training/validation split: 80/20
