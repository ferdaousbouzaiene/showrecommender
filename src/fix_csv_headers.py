# fix_csv_headers.py - Fix CSV Header Issues
import pandas as pd
import shutil
import os

def fix_csv_headers():
    """Fix the CSV header parsing issues"""
    
    print("🔧 FIXING CSV HEADER ISSUES")
    print("=" * 50)
    
    # Backup original files
    if os.path.exists('data/movies.csv'):
        shutil.copy('data/movies.csv', 'data/movies_backup.csv')
        print("✅ Backed up movies.csv")
    
    if os.path.exists('data/ratings.csv'):
        shutil.copy('data/ratings.csv', 'data/ratings_backup.csv')
        print("✅ Backed up ratings.csv")
    
    # Fix movies data
    try:
        print("\n📽️ Fixing movies data...")
        
        # Read without headers first
        movies_raw = pd.read_csv('data/movies.csv', header=None, low_memory=False)
        print(f"Raw movies shape: {movies_raw.shape}")
        print(f"First row: {movies_raw.iloc[0].tolist()[:5]}...")  # Show first 5 values
        
        # The first row seems to contain the actual header
        # Let's extract it and use it as column names
        if str(movies_raw.iloc[0, 0]).startswith('adult,belongs_to_collection'):
            print("Detected header row as first data row")
            # Split the first row to get proper column names
            header_string = str(movies_raw.iloc[0, 0])
            column_names = header_string.split(',')[:31]  # Take first 31 columns
            
            # Read again, skipping the malformed first row
            movies_df = pd.read_csv('data/movies.csv', skiprows=1, header=None, low_memory=False)
            
            # Assign proper column names
            if len(column_names) == len(movies_df.columns):
                movies_df.columns = column_names
            else:
                # Use the detected column structure from your diagnostic
                movies_df.columns = ['id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + \
                                  [f'genre_{i}' for i in range(19)] + \
                                  ['overview', 'genres', 'keywords', 'cast', 'crew', 'vote_average', 'vote_count']
        else:
            # Use your detected structure
            movies_df = movies_raw.copy()
            movies_df.columns = ['id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + \
                              [f'genre_{i}' for i in range(19)] + \
                              ['overview', 'genres', 'keywords', 'cast', 'crew', 'vote_average', 'vote_count']
        
        print(f"Fixed movies shape: {movies_df.shape}")
        print(f"Movies columns: {list(movies_df.columns)[:10]}...")
        
        # Save fixed movies file
        movies_df.to_csv('data/movies_fixed.csv', index=False)
        print("✅ Saved movies_fixed.csv")
        
    except Exception as e:
        print(f"❌ Error fixing movies: {e}")
        return False
    
    # Fix ratings data
    try:
        print("\n⭐ Fixing ratings data...")
        
        # Read without headers first
        ratings_raw = pd.read_csv('data/ratings.csv', header=None, low_memory=False)
        print(f"Raw ratings shape: {ratings_raw.shape}")
        print(f"First row: {ratings_raw.iloc[0].tolist()}")
        
        # Check if first row contains headers as data
        if str(ratings_raw.iloc[0, 0]) == 'userId,movieId,rating,timestamp':
            print("Detected header row as first data row")
            # Skip the malformed first row
            ratings_df = pd.read_csv('data/ratings.csv', skiprows=1, header=None, low_memory=False)
            ratings_df.columns = ['userId', 'movieId', 'rating', 'timestamp']
        else:
            # Use the raw data with proper column names
            ratings_df = ratings_raw.copy()
            ratings_df.columns = ['userId', 'movieId', 'rating', 'timestamp']
        
        print(f"Fixed ratings shape: {ratings_df.shape}")
        print(f"Ratings columns: {list(ratings_df.columns)}")
        print(f"Sample data:")
        print(ratings_df.head(3))
        
        # Save fixed ratings file
        ratings_df.to_csv('data/ratings_fixed.csv', index=False)
        print("✅ Saved ratings_fixed.csv")
        
    except Exception as e:
        print(f"❌ Error fixing ratings: {e}")
        return False
    
    # Replace original files with fixed versions
    try:
        shutil.move('data/movies_fixed.csv', 'data/movies.csv')
        shutil.move('data/ratings_fixed.csv', 'data/ratings.csv')
        print("\n✅ Replaced original files with fixed versions")
        return True
        
    except Exception as e:
        print(f"❌ Error replacing files: {e}")
        return False

def test_fixed_data():
    """Test if the fixed data loads correctly"""
    
    print("\n🧪 TESTING FIXED DATA")
    print("=" * 30)
    
    try:
        # Test movies
        movies = pd.read_csv('data/movies.csv', low_memory=False)
        print(f"✅ Movies loaded: {movies.shape}")
        print(f"   Sample title: {movies['title'].iloc[1] if len(movies) > 1 else 'N/A'}")
        print(f"   Valid IDs: {movies['id'].notna().sum()}")
        
        # Test ratings  
        ratings = pd.read_csv('data/ratings.csv', low_memory=False)
        print(f"✅ Ratings loaded: {ratings.shape}")
        print(f"   Sample rating: {ratings['rating'].iloc[1] if len(ratings) > 1 else 'N/A'}")
        print(f"   Valid ratings: {ratings['rating'].notna().sum()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = fix_csv_headers()
    if success:
        test_success = test_fixed_data()
        if test_success:
            print("\n🎉 DATA SUCCESSFULLY FIXED!")
            print("Now run: python setup_and_run.py")
        else:
            print("\n❌ Data test failed")
    else:
        print("\n❌ Fix failed")