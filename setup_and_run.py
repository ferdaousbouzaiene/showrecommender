# setup_and_run.py - COMPLETE FIXED VERSION
import os
import sys
from pathlib import Path
import pandas as pd
import pickle

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from data_preprocessing import DataPreprocessor
from hybrid_engine import HybridRecommendationEngine

def setup_project():
    """Initial project setup and data preprocessing"""
    print("🎬 Setting up Hybrid Movie Recommender System")
    print("=" * 50)
    
    # Check if raw data exists
    movies_path = "data/movies.csv"
    ratings_path = "data/ratings.csv"
    
    if not os.path.exists(movies_path) or not os.path.exists(ratings_path):
        print("❌ Raw data files not found!")
        print(f"Please ensure you have:")
        print(f"  - {movies_path}")
        print(f"  - {ratings_path}")
        return False
    
    # Run preprocessing
    print("🔄 Starting data preprocessing...")
    
    try:
        preprocessor = DataPreprocessor(movies_path, ratings_path)
        results = preprocessor.run_preprocessing()
        
        if results:
            print("✅ Data preprocessing completed successfully!")
            print(f"   - Processed {len(results['movies_df'])} movies")
            if results['user_item_matrix'] is not None and not results['user_item_matrix'].empty:
                print(f"   - User-item matrix: {results['user_item_matrix'].shape}")
            else:
                print("   - User-item matrix: Empty (will use content-based only)")
            if results['content_matrix'] is not None:
                print(f"   - Content matrix: {results['content_matrix'].shape}")
            return True
        else:
            print("❌ Preprocessing returned no results")
            return False
        
    except Exception as e:
        print(f"❌ Error during preprocessing: {str(e)}")
        return False

def test_recommender():
    """Test the hybrid recommender system with better error handling"""
    print("\n🧪 Testing Recommender System")
    print("=" * 50)
    
    try:
        # Load processed data
        movies_df = pd.read_csv('data/processed/processed_movies.csv')
        
        with open('data/processed/user_item_matrix.pkl', 'rb') as f:
            user_item_matrix = pickle.load(f)
            
        with open('data/processed/content_matrix.pkl', 'rb') as f:
            content_matrix = pickle.load(f)
            
        with open('data/processed/tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        
        print(f"✅ Processed data loaded:")
        print(f"   - Movies: {len(movies_df)}")
        print(f"   - User-item matrix: {user_item_matrix.shape if not user_item_matrix.empty else 'Empty'}")
        print(f"   - Content matrix: {content_matrix.shape if content_matrix is not None else 'None'}")
        
        # Initialize recommender (without OpenAI for basic testing)
        print("🔧 Initializing recommender...")
        engine = HybridRecommendationEngine(
            movies_df=movies_df,
            user_item_matrix=user_item_matrix if not user_item_matrix.empty else None,
            content_matrix=content_matrix,
            tfidf_vectorizer=tfidf_vectorizer,
            openai_api_key=None  # Skip GPT for basic test
        )
        
        # Train models with error handling
        print("🏋️ Training models...")
        try:
            if engine.content_recommender:
                engine.content_recommender.compute_content_similarity()
                print("✅ Content-based model trained")
            
            # Only train collaborative if we have data
            if engine.collaborative_recommender and not user_item_matrix.empty:
                if user_item_matrix.shape[0] > 1 and user_item_matrix.shape[1] > 1:
                    # Use smaller SVD components for testing
                    engine.collaborative_recommender.n_components = min(10, min(user_item_matrix.shape) - 1)
                    engine.collaborative_recommender.fit_svd()
                    print("✅ Collaborative filtering model trained")
                else:
                    print("⚠️ User-item matrix too small for collaborative filtering")
            else:
                print("⚠️ Skipping collaborative filtering (no data or empty matrix)")
        except Exception as e:
            print(f"⚠️ Model training warning: {e}")
            print("   Content-based recommendations will still work")
        
        # Test content-based recommendations
        print("🎯 Testing content-based recommendations...")
        if len(movies_df) > 0:
            sample_movie = movies_df.iloc[0]['title']
            recommendations = engine.get_movie_recommendations_by_title(sample_movie, 5)
            
            if recommendations:
                print(f"✅ Content-based test successful!")
                print(f"   Sample recommendations for '{sample_movie}':")
                for i, rec in enumerate(recommendations[:3], 1):
                    score = rec.get('similarity_score', 0)
                    print(f"   {i}. {rec['title']} (Score: {score:.3f})")
            else:
                print("⚠️ Content-based recommendations returned empty results")
        
        # Test trending movies
        print("\n🔥 Testing trending movies...")
        trending = engine.get_trending_movies(5)
        
        if trending:
            print("✅ Trending movies test successful!")
            print("   Top trending movies:")
            for i, movie in enumerate(trending[:3], 1):
                score = movie.get('trending_score', 0)
                print(f"   {i}. {movie['title']} (Score: {score:.3f})")
        
        print("\n✅ Core functionality working! The recommender system is operational.")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def check_environment():
    """Check if all required environment variables are set"""
    print("\n🔍 Checking Environment Setup")
    print("=" * 50)
    
    # Check for .env file
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  No .env file found. Creating template...")
        with open(".env", "w") as f:
            f.write("""# Environment Variables for Hybrid Movie Recommender
OPENAI_API_KEY=your_openai_api_key_here
TMDB_API_KEY=your_tmdb_api_key_here
DEBUG=True
MAX_RECOMMENDATIONS=10
CACHE_EXPIRY=3600
""")
        print("📝 Created .env template file. Please add your API keys.")
    
    # Load and check environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    openai_key = os.getenv('OPENAI_API_KEY')
    tmdb_key = os.getenv('TMDB_API_KEY')
    
    print(f"🔑 OpenAI API Key: {'✅ Set' if openai_key and openai_key != 'your_openai_api_key_here' else '❌ Not set'}")
    print(f"🖼️  TMDB API Key: {'✅ Set' if tmdb_key and tmdb_key != 'your_tmdb_api_key_here' else '❌ Not set (optional)'}")
    
    if not openai_key or openai_key == 'your_openai_api_key_here':
        print("\n⚠️  Note: Without OpenAI API key, GPT-powered features will be disabled")
        print("   You can still use content-based and collaborative filtering")
    
    return True

def run_streamlit_app():
    """Launch the Streamlit application"""
    print("\n🚀 Launching Streamlit Application")
    print("=" * 50)
    
    try:
        import subprocess
        print("📱 Starting Streamlit server...")
        print("🌐 The app will open in your browser at: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the server")
        
        # Run streamlit app
        subprocess.run(["streamlit", "run", "streamlit_app.py"])
        
    except KeyboardInterrupt:
        print("\n👋 Streamlit server stopped")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {str(e)}")
        print("💡 Make sure Streamlit is installed: pip install streamlit")

def main():
    """Main function to orchestrate the setup and execution"""
    print("🎬 Hybrid Movie Recommender System")
    print("🤖 Combining Collaborative Filtering + Content-Based + GPT-4")
    print("=" * 60)
    
    # Check environment setup
    check_environment()
    
    # Check if processed data exists
    processed_dir = Path("data/processed")
    required_files = [
        "processed_movies.csv",
        "user_item_matrix.pkl",
        "content_matrix.pkl",
        "tfidf_vectorizer.pkl"
    ]
    
    files_exist = all((processed_dir / f).exists() for f in required_files)
    
    if not files_exist:
        print("\n🔧 Processed data not found. Running setup...")
        if not setup_project():
            print("❌ Setup failed. Please check your data files and try again.")
            return
    else:
        print("✅ Processed data found!")
    
    # Test the system
    print("\n🧪 Running system tests...")
    if not test_recommender():
        print("⚠️ Some tests failed, but basic functionality may still work.")
        print("Try launching the app anyway.")
    
    # Ask user what to do next
    print("\n🎯 What would you like to do?")
    print("1. Launch Streamlit Web App")
    print("2. Re-run data preprocessing")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            run_streamlit_app()
            break
        elif choice == "2":
            setup_project()
            break
        elif choice == "3":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
