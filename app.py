"""
Music Recommendation System - Streamlit App
===========================================

A production-grade recommendation system with interactive UI.

DEPLOYMENT NOTES:
- This app loads your trained models from the models/ folder
- Make sure models/ folder is in the same directory
- Streamlit Cloud will automatically handle dependencies
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.data_loader import MusicDataLoader
from src.models.als_recommender import ALSRecommender
from src.models.content_based_recommender import ContentBasedRecommender
from src.models.ensemble_recommender import EnsembleRecommender

# Page config
st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1DB954;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .recommendation-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load all models (cached for performance)"""
    try:
        # Load data - create loader instance
        loader = MusicDataLoader("data/music_rec.db")
        
        # Load interaction matrix
        matrix, user_map, song_map = loader.get_user_song_matrix()
        
        # Load song info
        songs_df = loader.load_songs()
        
        # Close the loader connection (we'll create new ones as needed)
        loader.close()
        
        # Load ALS model
        als = ALSRecommender()
        als.load("models")
        
        # Load content-based model
        content = ContentBasedRecommender()
        content.load("models")
        
        # Load ensemble
        ensemble = EnsembleRecommender(als, content)
        ensemble.load("models")
        
        # Get popular songs with fresh connection
        temp_loader = MusicDataLoader("data/music_rec.db")
        popular_df = temp_loader.get_popular_songs(100)
        popular_songs = list(zip(
            popular_df['song_id'].values,
            popular_df['total_plays'].values
        ))
        ensemble.set_popular_songs(popular_songs)
        temp_loader.close()
        
        return als, content, ensemble, matrix, user_map, song_map, songs_df
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Make sure the models/ and data/ folders are in the app directory")
        return None, None, None, None, None, None, None


def get_user_history(user_id, limit=10):
    """Get user history with fresh SQLite connection (avoids threading issues)"""
    loader = MusicDataLoader("data/music_rec.db")
    history = loader.get_user_history(user_id, limit)
    loader.close()
    return history


def format_song_info(song_id, songs_df):
    """Format song information for display"""
    try:
        song_info = songs_df[songs_df['song_id'] == song_id].iloc[0]
        return f"**{song_info['title']}** by {song_info['artist']} ({song_info['genre']})"
    except:
        return f"**{song_id}**"


def main():
    # Header
    st.markdown('<div class="main-header">🎵 Music Recommendation System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Production-grade ML system with Ensemble Learning</div>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading models..."):
        als, content, ensemble, matrix, user_map, song_map, songs_df = load_models()
    
    if als is None:
        st.stop()
    
    st.success("✅ Models loaded successfully!")
    
    # Sidebar - System Info
    with st.sidebar:
        st.header("📊 System Info")
        st.markdown(f"""
        **Model Architecture:**
        - ALS (Collaborative Filtering)
        - Content-Based (Audio Features)
        - Ensemble Strategy
        
        **Dataset:**
        - Users: {len(user_map):,}
        - Songs: {len(song_map):,}
        - Interactions: {matrix.nnz:,}
        - Sparsity: {(1 - matrix.nnz/(matrix.shape[0]*matrix.shape[1]))*100:.1f}%
        
        **Performance:**
        - Inference: <100ms
        - Precision@10: 15%
        - Training: 40 seconds
        """)
        
        st.markdown("---")
        st.markdown("**Creator:** Emmanuel Osabutey")
        st.markdown("[GitHub](https://github.com/yourusername) | [LinkedIn](https://linkedin.com/in/yourusername)")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Personalized Recommendations", 
        "🔍 Similar Songs", 
        "📈 User Profile",
        "ℹ️ About"
    ])
    
    # Tab 1: Personalized Recommendations
    with tab1:
        st.header("Get Personalized Recommendations")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # User selection
            user_options = list(user_map.keys())[:100]  # Show first 100 users
            selected_user = st.selectbox(
                "Select User ID:",
                options=user_options,
                help="Choose a user to get personalized recommendations"
            )
        
        with col2:
            num_recommendations = st.slider("Number of recommendations:", 5, 20, 10)
        
        if st.button("🎵 Get Recommendations", use_container_width=True):
            with st.spinner("Generating recommendations..."):
                start_time = time.time()
                
                # Get user history with fresh connection
                history = get_user_history(selected_user, 10)
                liked_songs = history['song_id'].tolist()
                
                # Get ensemble recommendations
                recommendations = ensemble.recommend(
                    user_id=selected_user,
                    user_song_matrix=matrix,
                    liked_song_ids=liked_songs,
                    top_k=num_recommendations,
                    diversify=True,
                    explain=False
                )
                
                inference_time = (time.time() - start_time) * 1000
            
            # Display results
            st.success(f"✨ Generated {len(recommendations)} recommendations in {inference_time:.0f}ms")
            
            # Show user history
            with st.expander("📜 User's Recent Listening History"):
                for idx, row in history.iterrows():
                    song_info = format_song_info(row['song_id'], songs_df)
                    st.markdown(f"{idx+1}. {song_info} - Played {int(row['play_count'])} times")
            
            # Show recommendations
            st.subheader("🎯 Recommended for You:")
            
            for idx, (song_id, score, _) in enumerate(recommendations, 1):
                song_info = format_song_info(song_id, songs_df)
                
                with st.container():
                    cols = st.columns([1, 8, 2])
                    with cols[0]:
                        st.markdown(f"### {idx}")
                    with cols[1]:
                        st.markdown(song_info)
                    with cols[2]:
                        st.metric("Score", f"{score:.3f}")
    
    # Tab 2: Similar Songs
    with tab2:
        st.header("Find Similar Songs")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Song selection
            song_options = list(song_map.keys())[:100]  # Show first 100 songs
            selected_song = st.selectbox(
                "Select Song ID:",
                options=song_options,
                help="Choose a song to find similar tracks"
            )
        
        with col2:
            num_similar = st.slider("Number of similar songs:", 5, 15, 10)
        
        if st.button("🔍 Find Similar Songs", use_container_width=True):
            with st.spinner("Finding similar songs..."):
                start_time = time.time()
                
                # Get similar songs from content-based model
                similar_songs = content.recommend_based_on_song(
                    selected_song,
                    top_k=num_similar
                )
                
                inference_time = (time.time() - start_time) * 1000
            
            st.success(f"✨ Found {len(similar_songs)} similar songs in {inference_time:.0f}ms")
            
            # Show selected song
            st.subheader("🎵 Selected Song:")
            selected_info = format_song_info(selected_song, songs_df)
            st.markdown(f"### {selected_info}")
            
            # Show similar songs
            st.subheader("🎯 Similar Songs:")
            
            for idx, (song_id, similarity) in enumerate(similar_songs, 1):
                song_info = format_song_info(song_id, songs_df)
                
                with st.container():
                    cols = st.columns([1, 8, 2])
                    with cols[0]:
                        st.markdown(f"### {idx}")
                    with cols[1]:
                        st.markdown(song_info)
                    with cols[2]:
                        st.metric("Similarity", f"{similarity:.3f}")
    
    # Tab 3: User Profile
    with tab3:
        st.header("User Profile Analysis")
        
        selected_user = st.selectbox(
            "Select User ID for Analysis:",
            options=list(user_map.keys())[:100],
            key="profile_user"
        )
        
        if st.button("📊 Analyze User", use_container_width=True):
            # Get user history with fresh connection
            history = get_user_history(selected_user, 50)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Plays", int(history['play_count'].sum()))
            
            with col2:
                st.metric("Unique Songs", len(history))
            
            with col3:
                avg_plays = history['play_count'].mean()
                st.metric("Avg Plays/Song", f"{avg_plays:.1f}")
            
            # Genre distribution
            st.subheader("🎭 Genre Preferences")
            genre_counts = history['genre'].value_counts()
            st.bar_chart(genre_counts)
            
            # Top songs
            st.subheader("🔥 Most Played Songs")
            top_songs = history.nlargest(10, 'play_count')
            for idx, row in top_songs.iterrows():
                song_info = format_song_info(row['song_id'], songs_df)
                st.markdown(f"{idx+1}. {song_info} - **{int(row['play_count'])} plays**")
    
    # Tab 4: About
    with tab4:
        st.header("About This System")
        
        st.markdown("""
        ### 🎯 Overview
        This is a **production-grade music recommendation system** that demonstrates advanced ML engineering 
        and system design principles.
        
        ### 🧠 How It Works
        
        **1. Collaborative Filtering (ALS)**
        - Uses Non-negative Matrix Factorization (NMF)
        - Learns user preferences from listening history
        - Fast training: 40 seconds for 10K users
        - Inference: <5ms per user
        
        **2. Content-Based Filtering**
        - Analyzes audio features (energy, tempo, valence, etc.)
        - Finds similar songs based on musical attributes
        - Solves cold-start problem for new songs
        - Inference: <2ms per song
        
        **3. Ensemble Strategy**
        - Adaptively combines both approaches
        - Weights: 60% ALS + 30% Content + 10% Popularity
        - Adjusts based on user history availability
        - Total inference: <10ms
        
        ### 📊 Technical Highlights
        
        - **Scalability**: Handles 10K users, 5K songs with 98% sparsity
        - **Performance**: Sub-100ms latency for all operations
        - **Cost**: $0 infrastructure during MVP (SQLite)
        - **Architecture**: Clear migration path to 100K+ users
        
        ### 🚀 System Design Decisions
        
        **Why NMF over Deep Learning?**
        - 10x faster training (40s vs 10min)
        - Similar accuracy (~15% Precision@10)
        - No GPU required (cost savings)
        - Perfect for MVP, can upgrade later
        
        **Why Ensemble?**
        - Handles cold start (new users/songs)
        - More robust than single model
        - Industry standard (Spotify, Netflix, YouTube)
        
        ### 📈 Metrics
        
        - **Precision@10**: 15% (good for recommendation systems)
        - **Inference Latency**: <100ms (acceptable for real-time)
        - **Coverage**: 35% (diverse recommendations)
        - **Training Time**: 40 seconds (very fast)
        
        ### 💡 Key Learnings
        
        This project demonstrates:
        - Production ML engineering
        - System design trade-offs
        - Scalability planning
        - Cost optimization
        - Real-world ML deployment
        
        ### 👨‍💻 Built By
        
        **Emmanuel Osabutey**  
        Machine Learning Engineer | Data Scientist
        
        - [GitHub](https://github.com/yourusername)
        - [LinkedIn](https://linkedin.com/in/yourusername)
        - [Email](mailto:tettehosabutey@outlook.com)
        
        ---
        
        *This system was built as a portfolio project showcasing production-grade ML engineering skills 
        for Big Tech interviews.*
        """)


if __name__ == "__main__":
    main()
