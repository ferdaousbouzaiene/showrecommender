# streamlit_app.py - FIXED VERSION
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from dotenv import load_dotenv
import sys
from pathlib import Path

# Fix import path issues
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.append(str(src_dir))

# Now import with absolute imports
from hybrid_engine import HybridRecommendationEngine
import requests
from PIL import Image
from io import BytesIO

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Hybrid Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #e50914;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .movie-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f9f9f9;
    }
    
    .recommendation-source {
        background-color: #e1f5fe;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .gpt-explanation {
        background-color: #f3e5f5;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #9c27b0;
        margin: 0.5rem 0;
    }
    
    .score-badge {
        background-color: #4caf50;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load preprocessed data with caching"""
    try:
        # Load processed data
        movies_df = pd.read_csv('data/processed/processed_movies.csv')
        
        with open('data/processed/user_item_matrix.pkl', 'rb') as f:
            user_item_matrix = pickle.load(f)
            
        with open('data/processed/content_matrix.pkl', 'rb') as f:
            content_matrix = pickle.load(f)
            
        with open('data/processed/tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
            
        return movies_df, user_item_matrix, content_matrix, tfidf_vectorizer
        
    except FileNotFoundError as e:
        st.error(f"Data files not found: {e}")
        st.info("Please run the data preprocessing script first.")
        return None, None, None, None

@st.cache_resource
def initialize_recommender(_movies_df, _user_item_matrix, _content_matrix, _tfidf_vectorizer):
    """Initialize the hybrid recommendation engine with caching"""
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    engine = HybridRecommendationEngine(
        movies_df=_movies_df,
        user_item_matrix=_user_item_matrix,
        content_matrix=_content_matrix,
        tfidf_vectorizer=_tfidf_vectorizer,
        openai_api_key=openai_api_key
    )
    
    engine.train_models()
    return engine

def fetch_movie_poster(movie_title, api_key=None):
    """Fetch movie poster from TMDB API"""
    if not api_key:
        return None
        
    try:
        # Search for movie
        search_url = f"https://api.themoviedb.org/3/search/movie"
        params = {
            'api_key': api_key,
            'query': movie_title
        }
        
        response = requests.get(search_url, params=params)
        if response.status_code == 200:
            results = response.json().get('results', [])
            if results:
                poster_path = results[0].get('poster_path')
                if poster_path:
                    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                    return poster_url
                    
    except Exception as e:
        st.warning(f"Could not fetch poster for {movie_title}: {str(e)}")
    
    return None

def display_movie_card(movie, show_poster=True, api_key=None):
    """Display a movie recommendation card"""
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if show_poster and api_key:
            poster_url = fetch_movie_poster(movie['title'], api_key)
            if poster_url:
                try:
                    response = requests.get(poster_url)
                    img = Image.open(BytesIO(response.content))
                    st.image(img, width=150)
                except:
                    st.write("🎬")
            else:
                st.write("🎬")
        else:
            st.write("🎬")
    
    with col2:
        st.markdown(f"**{movie['title']}**")
        
        # Display release date and rating
        col_a, col_b = st.columns(2)
        with col_a:
            release_year = movie.get('release_date', '')[:4] if movie.get('release_date') else 'Unknown'
            st.write(f"**Year:** {release_year}")
        with col_b:
            rating = movie.get('vote_average', 0)
            st.write(f"**Rating:** ⭐ {rating:.1f}")
        
        # Display genres
        genres = movie.get('genres', 'Unknown')
        st.write(f"**Genres:** {genres}")
        
        # Display overview
        overview = movie.get('overview', 'No overview available.')
        if len(overview) > 200:
            overview = overview[:200] + "..."
        st.write(f"**Plot:** {overview}")
        
        # Display recommendation sources if available
        if 'recommendation_sources' in movie:
            sources = movie['recommendation_sources']
            for source in sources:
                st.markdown(f'<span class="recommendation-source">{source}</span>', unsafe_allow_html=True)
        
        # Display hybrid score if available
        if 'hybrid_score' in movie:
            score = movie['hybrid_score']
            st.markdown(f'<span class="score-badge">Score: {score:.2f}</span>', unsafe_allow_html=True)
        
        # Display GPT explanation if available
        if 'gpt_reason' in movie and movie['gpt_reason']:
            st.markdown(f'<div class="gpt-explanation"><strong>AI Insight:</strong> {movie["gpt_reason"]}</div>', 
                       unsafe_allow_html=True)
        elif 'gpt_explanation' in movie and movie['gpt_explanation']:
            st.markdown(f'<div class="gpt-explanation"><strong>AI Insight:</strong> {movie["gpt_explanation"]}</div>', 
                       unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🎬 Hybrid Movie Recommender</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Discover your next favorite movie using AI-powered recommendations</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading movie database..."):
        movies_df, user_item_matrix, content_matrix, tfidf_vectorizer = load_data()
    
    if movies_df is None:
        st.stop()
    
    # Initialize recommender
    with st.spinner("Initializing recommendation engine..."):
        recommender = initialize_recommender(movies_df, user_item_matrix, content_matrix, tfidf_vectorizer)
    
    # API keys
    tmdb_api_key = os.getenv('TMDB_API_KEY')
    
    # Sidebar
    st.sidebar.header("Recommendation Options")
    
    recommendation_type = st.sidebar.selectbox(
        "Choose recommendation method:",
        ["Movie-Based", "Natural Language", "Trending Movies", "Random Discovery"]
    )
    
    show_posters = st.sidebar.checkbox("Show movie posters", value=bool(tmdb_api_key))
    if show_posters and not tmdb_api_key:
        st.sidebar.warning("TMDB API key required for posters")
        show_posters = False
    
    num_recommendations = st.sidebar.slider("Number of recommendations", 1, 20, 10)
    
    # Main content
    if recommendation_type == "Movie-Based":
        st.header("🎯 Find Similar Movies")
        st.write("Enter a movie you liked, and we'll find similar ones for you!")
        
        # Movie search
        search_query = st.text_input("Search for a movie:", placeholder="e.g., The Dark Knight, Inception, Pulp Fiction")
        
        if search_query:
            # Search for movies
            search_results = recommender.search_movies(search_query, 5)
            
            if search_results:
                st.subheader("Select a movie:")
                selected_movie = st.selectbox(
                    "Choose from search results:",
                    options=search_results,
                    format_func=lambda x: f"{x['title']} ({x.get('release_date', '')[:4] if x.get('release_date') else 'Unknown'})"
                )
                
                if st.button("Get Recommendations", type="primary"):
                    with st.spinner("Finding similar movies..."):
                        recommendations = recommender.get_movie_recommendations_by_title(
                            selected_movie['title'], num_recommendations
                        )
                    
                    if recommendations:
                        st.success(f"Found {len(recommendations)} recommendations based on '{selected_movie['title']}'")
                        
                        for i, movie in enumerate(recommendations, 1):
                            with st.expander(f"{i}. {movie['title']} (Score: {movie.get('similarity_score', 0):.2f})"):
                                display_movie_card(movie, show_posters, tmdb_api_key)
                    else:
                        st.warning("No recommendations found for this movie.")
            else:
                st.info("No movies found. Try a different search term.")
    
    elif recommendation_type == "Natural Language":
        st.header("🤖 AI-Powered Recommendations")
        st.write("Describe what kind of movie you're in the mood for, and our AI will suggest perfect matches!")
        
        # Check if GPT is available
        if not recommender.gpt_recommender:
            st.warning("OpenAI API key required for AI recommendations. Please set OPENAI_API_KEY in your .env file.")
        else:
            preferences = st.text_area(
                "What kind of movie are you looking for?",
                placeholder="e.g., I want a mind-bending sci-fi thriller like Inception, or a heartwarming romantic comedy, or dark psychological horror movies...",
                height=100
            )
            
            if preferences and st.button("Get AI Recommendations", type="primary"):
                with st.spinner("AI is analyzing your preferences..."):
                    result = recommender.get_recommendations_by_preferences(preferences, num_recommendations)
                
                if result['success'] and result['recommendations']:
                    st.success(f"Found {len(result['recommendations'])} AI-powered recommendations!")
                    
                    if result['explanation']:
                        st.info(f"**AI Strategy:** {result['explanation']}")
                    
                    for i, movie in enumerate(result['recommendations'], 1):
                        with st.expander(f"{i}. {movie['title']}"):
                            display_movie_card(movie, show_posters, tmdb_api_key)
                else:
                    st.error(f"Could not generate recommendations: {result.get('error', 'Unknown error')}")
    
    elif recommendation_type == "Trending Movies":
        st.header("🔥 Trending Movies")
        st.write("Discover popular and highly-rated movies!")
        
        if st.button("Show Trending Movies", type="primary"):
            with st.spinner("Finding trending movies..."):
                trending = recommender.get_trending_movies(num_recommendations)
            
            st.success(f"Here are {len(trending)} trending movies:")
            
            for i, movie in enumerate(trending, 1):
                with st.expander(f"{i}. {movie['title']} (Trending Score: {movie.get('trending_score', 0):.2f})"):
                    display_movie_card(movie, show_posters, tmdb_api_key)
    
    elif recommendation_type == "Random Discovery":
        st.header("🎲 Random Movie Discovery")
        st.write("Feeling adventurous? Let us surprise you with random movie picks!")
        
        if st.button("Discover Random Movies", type="primary"):
            with st.spinner("Picking random movies..."):
                random_movies = recommender.get_random_recommendations(num_recommendations)
            
            st.success(f"Here are {len(random_movies)} random movie discoveries:")
            
            for i, movie in enumerate(random_movies, 1):
                with st.expander(f"{i}. {movie['title']}"):
                    display_movie_card(movie, show_posters, tmdb_api_key)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; margin-top: 2rem;">
            <p>Powered by Collaborative Filtering, Content-Based Filtering, and OpenAI GPT-4</p>
            <p>Built with ❤️ using Streamlit, scikit-learn, and pandas</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()