"""
Music Recommendation System - Training Script
==============================================

Run from project root: python train.py --generate-data

Simple, synchronous training pipeline for Part 1.
"""

import os
import sys
import time
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from loguru import logger

# Import our modules
from src.data.data_generator import MusicDataGenerator
from src.data.data_loader import MusicDataLoader
from src.models.als_recommender import ALSRecommender
from src.models.content_based_recommender import ContentBasedRecommender
from src.models.ensemble_recommender import EnsembleRecommender


def generate_data(n_users=10000, n_songs=5000, n_interactions=100000):
    """Stage 1: Generate synthetic data"""
    logger.info("=" * 60)
    logger.info("STAGE 1: DATA GENERATION")
    logger.info("=" * 60)
    
    generator = MusicDataGenerator(
        n_users=n_users,
        n_songs=n_songs,
        n_interactions=n_interactions,
        output_dir="data"
    )
    
    users_df, songs_df, interactions_df = generator.generate_all()
    
    logger.info(f"✓ Generated {len(users_df)} users, {len(songs_df)} songs")
    return True


def load_and_split_data():
    """Stage 2: Load and split data"""
    logger.info("=" * 60)
    logger.info("STAGE 2: DATA LOADING & SPLITTING")
    logger.info("=" * 60)
    
    loader = MusicDataLoader("data/music_rec.db")
    
    # Load all data
    users_df = loader.load_users()
    songs_df = loader.load_songs()
    interactions_df = loader.load_interactions()
    
    logger.info(f"Loaded {len(interactions_df)} interactions")
    
    # Split train/test
    train_df, test_df = train_test_split(
        interactions_df, 
        test_size=0.2, 
        random_state=42
    )
    
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Get user-song matrix
    train_matrix, user_map, song_map = loader.get_user_song_matrix()
    
    # Get song features
    feature_cols = [
        'acousticness_norm', 'danceability_norm', 'energy_norm',
        'instrumentalness_norm', 'liveness_norm', 'speechiness_norm',
        'valence_norm', 'tempo_norm'
    ]
    song_features = songs_df[feature_cols].values
    song_ids = songs_df['song_id'].values
    genres = songs_df['genre'].values
    
    loader.close()
    
    return train_matrix, user_map, song_map, song_features, song_ids, genres


def train_als(train_matrix, user_map, song_map):
    """Stage 3: Train ALS model"""
    logger.info("=" * 60)
    logger.info("STAGE 3: TRAINING ALS MODEL")
    logger.info("=" * 60)
    
    als = ALSRecommender(
        factors=64,
        iterations=15,
        regularization=0.01,
        alpha=40,
        use_gpu=False
    )
    
    als.fit(train_matrix, user_map, song_map)
    
    logger.info("✓ ALS training complete")
    return als


def train_content(song_features, song_ids, genres):
    """Stage 4: Train content-based model"""
    logger.info("=" * 60)
    logger.info("STAGE 4: TRAINING CONTENT-BASED MODEL")
    logger.info("=" * 60)
    
    content = ContentBasedRecommender(
        similarity_threshold=0.7,
        use_genre=True
    )
    
    content.fit(song_features, song_ids, genres)
    
    logger.info("✓ Content-based training complete")
    return content


def create_ensemble(als, content):
    """Stage 5: Create ensemble"""
    logger.info("=" * 60)
    logger.info("STAGE 5: CREATING ENSEMBLE")
    logger.info("=" * 60)
    
    # Get popular songs
    loader = MusicDataLoader("data/music_rec.db")
    popular_df = loader.get_popular_songs(top_k=100)
    popular_songs = list(zip(
        popular_df['song_id'].values,
        popular_df['total_plays'].values
    ))
    loader.close()
    
    ensemble = EnsembleRecommender(
        als_recommender=als,
        content_recommender=content,
        als_weight=0.6,
        content_weight=0.3,
        popularity_weight=0.1
    )
    
    ensemble.set_popular_songs(popular_songs)
    
    logger.info("✓ Ensemble created")
    return ensemble


def save_models(als, content, ensemble):
    """Stage 6: Save all models"""
    logger.info("=" * 60)
    logger.info("STAGE 6: SAVING MODELS")
    logger.info("=" * 60)
    
    # Create models directory
    Path("models").mkdir(exist_ok=True)
    
    # Save models
    als.save("models")
    content.save("models")
    ensemble.save("models")
    
    logger.info("✓ All models saved to models/")


def test_models(als, content, ensemble, train_matrix, user_map):
    """Quick test of models"""
    logger.info("=" * 60)
    logger.info("TESTING MODELS")
    logger.info("=" * 60)
    
    # Get a sample user
    sample_user = list(user_map.keys())[0]
    
    # Test ALS
    als_recs = als.recommend_for_user(sample_user, train_matrix, top_k=5)
    logger.info(f"\nALS recommendations for {sample_user}:")
    for song_id, score in als_recs[:3]:
        logger.info(f"  {song_id}: {score:.3f}")
    
    # Test Content
    sample_song = list(content.song_id_to_idx.keys())[0]
    content_recs = content.recommend_based_on_song(sample_song, top_k=5)
    logger.info(f"\nContent-based similar to {sample_song}:")
    for song_id, score in content_recs[:3]:
        logger.info(f"  {song_id}: {score:.3f}")
    
    logger.info("\n✓ Models working correctly!")


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description="Train music recommendation models")
    parser.add_argument("--generate-data", action="store_true", help="Generate new data")
    parser.add_argument("--n-users", type=int, default=10000, help="Number of users")
    parser.add_argument("--n-songs", type=int, default=5000, help="Number of songs")
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    logger.info("""
╔══════════════════════════════════════════════════════╗
║  🎵 Music Recommendation System - Training Pipeline  ║
║                                                      ║
║  Part 1: Core ML System                             ║
║  - ALS Collaborative Filtering                      ║
║  - Content-Based Recommendations                     ║
║  - Ensemble System                                   ║
╚══════════════════════════════════════════════════════╝
    """)
    
    start_time = time.time()
    
    try:
        # Stage 1: Generate data (if requested)
        if args.generate_data:
            generate_data(args.n_users, args.n_songs)
        
        # Stage 2: Load and split
        train_matrix, user_map, song_map, song_features, song_ids, genres = load_and_split_data()
        
        # Stage 3: Train ALS
        als = train_als(train_matrix, user_map, song_map)
        
        # Stage 4: Train content-based
        content = train_content(song_features, song_ids, genres)
        
        # Stage 5: Create ensemble
        ensemble = create_ensemble(als, content)
        
        # Stage 6: Save models
        save_models(als, content, ensemble)
        
        # Test
        test_models(als, content, ensemble, train_matrix, user_map)
        
        # Summary
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("🎉 TRAINING COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        logger.info("\nModels saved to:")
        logger.info("  - models/als_model.pkl")
        logger.info("  - models/content_based_model.pkl")
        logger.info("  - models/ensemble_config.pkl")
        logger.info("\nNext steps:")
        logger.info("  1. Test: python src/models/als_recommender.py")
        logger.info("  2. Read code comments for system design details")
        logger.info("  3. Ready for Part 2 (API + Docker)!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
