# diagnose_data.py - Data Format Diagnostic Script
import pandas as pd
import numpy as np

def diagnose_data():
    """Diagnose data format issues"""
    
    print("🔍 DIAGNOSING YOUR DATA FORMAT")
    print("=" * 50)
    
    try:
        # Load and examine movies data
        print("\n📽️ MOVIES DATA ANALYSIS:")
        movies_df = pd.read_csv('data/movies.csv', low_memory=False)
        print(f"Shape: {movies_df.shape}")
        print(f"Columns: {list(movies_df.columns)}")
        print(f"First few rows:")
        print(movies_df.head(2))
        
        # Check for common ID columns
        potential_id_cols = [col for col in movies_df.columns if 'id' in col.lower()]
        print(f"Potential ID columns: {potential_id_cols}")
        
        # Check data types
        print(f"Data types:")
        print(movies_df.dtypes)
        
    except Exception as e:
        print(f"Error loading movies data: {e}")
        return False
    
    try:
        # Load and examine ratings data  
        print("\n⭐ RATINGS DATA ANALYSIS:")
        ratings_df = pd.read_csv('data/ratings.csv', low_memory=False)
        print(f"Shape: {ratings_df.shape}")
        print(f"Columns: {list(ratings_df.columns)}")
        print(f"First few rows:")
        print(ratings_df.head(2))
        
        # Check for common columns
        potential_user_cols = [col for col in ratings_df.columns if 'user' in col.lower()]
        potential_movie_cols = [col for col in ratings_df.columns if any(word in col.lower() for word in ['movie', 'item', 'id'])]
        potential_rating_cols = [col for col in ratings_df.columns if any(word in col.lower() for word in ['rating', 'score'])]
        
        print(f"Potential user columns: {potential_user_cols}")
        print(f"Potential movie columns: {potential_movie_cols}")  
        print(f"Potential rating columns: {potential_rating_cols}")
        
        print(f"Data types:")
        print(ratings_df.dtypes)
        
    except Exception as e:
        print(f"Error loading ratings data: {e}")
        return False
    
    print("\n🔧 RECOMMENDATIONS:")
    
    # Check if we have the expected columns
    expected_movie_cols = ['id', 'title', 'overview', 'genres']
    expected_rating_cols = ['userId', 'movieId', 'rating']
    
    print(f"\nExpected movie columns: {expected_movie_cols}")
    missing_movie_cols = [col for col in expected_movie_cols if col not in movies_df.columns]
    if missing_movie_cols:
        print(f"❌ Missing movie columns: {missing_movie_cols}")
        # Suggest mappings
        for missing_col in missing_movie_cols:
            similar_cols = [col for col in movies_df.columns if missing_col.lower() in col.lower() or col.lower() in missing_col.lower()]
            if similar_cols:
                print(f"   Possible mapping for '{missing_col}': {similar_cols}")
    else:
        print("✅ All expected movie columns found")
    
    print(f"\nExpected rating columns: {expected_rating_cols}")
    missing_rating_cols = [col for col in expected_rating_cols if col not in ratings_df.columns]
    if missing_rating_cols:
        print(f"❌ Missing rating columns: {missing_rating_cols}")
        for missing_col in missing_rating_cols:
            similar_cols = [col for col in ratings_df.columns if missing_col.lower() in col.lower() or col.lower() in missing_col.lower()]
            if similar_cols:
                print(f"   Possible mapping for '{missing_col}': {similar_cols}")
    else:
        print("✅ All expected rating columns found")
    
    return True

if __name__ == "__main__":
    diagnose_data()