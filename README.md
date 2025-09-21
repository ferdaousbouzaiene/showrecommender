# 🎬 Hybrid Movie Recommender System

A sophisticated movie recommendation system that combines **Collaborative Filtering**, **Content-Based Filtering**, and **GPT-4 AI** to deliver personalized movie suggestions with intelligent explanations.

## ✨ Features

- **🤝 Collaborative Filtering**: Recommendations based on user behavior patterns and similar users
- **📝 Content-Based Filtering**: Suggestions using movie metadata (genres, cast, plot, keywords)
- **🤖 GPT-4 Integration**: Natural language movie recommendations with AI-powered explanations
- **⚡ Hybrid Scoring**: Combines multiple approaches for superior accuracy
- **🌐 Interactive Web App**: Beautiful Streamlit interface with movie posters
- **📊 Smart Analytics**: Trending movies, similarity scoring, and recommendation explanations

## 🛠️ Tech Stack

- **Python 3.8+**
- **Streamlit** - Web application framework
- **scikit-learn** - Machine learning algorithms
- **pandas & numpy** - Data manipulation
- **OpenAI GPT-4** - Natural language processing
- **TMDB API** - Movie posters and metadata (optional)

## 📁 Project Structure

```
hybrid-movie-recommender/
├── data/
│   ├── movies.csv                 # Raw movie metadata
│   ├── ratings.csv                # User ratings data
│   └── processed/                 # Processed data files
├── src/
│   ├── data_preprocessing.py      # Data processing pipeline
│   ├── collaborative_filtering.py # Collaborative filtering engine
│   ├── content_based.py           # Content-based filtering engine
│   ├── gpt_recommender.py         # GPT-4 integration
│   ├── hybrid_engine.py           # Main hybrid recommendation engine
│   └── utils.py                   # Utility functions
├── streamlit_app.py               # Web application
├── setup_and_run.py               # Setup and execution script
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variables template
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Clone and Install

```bash
git clone <repository-url>
cd hybrid-movie-recommender
pip install -r requirements.txt
```

### 2. Get Your Data

Download movie datasets from one of these sources:

**Option A: Kaggle TMDB Dataset**
- Visit: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
- Download and place files in `data/` folder

**Option B: MovieLens Dataset**
- Visit: https://grouplens.org/datasets/movielens/
- Use the 25M dataset for best results

**Required files:**
- `data/movies.csv` - Movie metadata
- `data/ratings.csv` - User ratings

### 3. Set Up API Keys

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
# Required for GPT-powered recommendations
OPENAI_API_KEY=your_openai_api_key_here

# Optional for movie posters
TMDB_API_KEY=your_tmdb_api_key_here
```

**Get API Keys:**
- **OpenAI**: https://platform.openai.com/api-keys
- **TMDB**: https://www.themoviedb.org/settings/api

### 4. Run the Application

```bash
python setup_and_run.py
```

This will:
1. ✅ Check your environment setup
2. 🔄 Process the data (first run only)
3. 🧪 Test the recommendation system
4. 🚀 Launch the Streamlit web application

## 🎯 How to Use

### Web Application Features

1. **🎬 Movie-Based Recommendations**
   - Search for a movie you liked
   - Get similar movies with explanations
   - Uses content-based filtering + GPT insights

2. **🤖 AI-Powered Natural Language**
   - Describe your mood: "I want a mind-bending thriller"
   - Get personalized recommendations with AI reasoning
   - Powered by GPT-4 with movie database context

3. **🔥 Trending Movies**
   - Discover popular and highly-rated films
   - Algorithm combines rating and vote count

4. **🎲 Random Discovery**
   - Serendipitous movie discovery
   - Great for breaking out of your comfort zone

### Python API Usage

```python
from src.hybrid_engine import HybridRecommendationEngine
import pandas as pd

# Load your data
movies_df = pd.read_csv('data/processed/processed_movies.csv')
# ... load other processed data

# Initialize engine
engine = HybridRecommendationEngine(
    movies_df=movies_df,
    user_item_matrix=user_item_matrix,
    content_matrix=content_matrix,
    tfidf_vectorizer=tfidf_vectorizer,
    openai_api_key=your_api_key
)

# Get recommendations
recommendations = engine.get_movie_recommendations_by_title("Inception", 10)

# Natural language recommendations
result = engine.get_recommendations_by_preferences(
    "I want a dark psychological thriller with great cinematography"
)
```

## 🧠 How It Works

### 1. Data Preprocessing
- Cleans and processes movie metadata
- Creates user-item interaction matrix
- Generates TF-IDF vectors from movie features
- Handles missing data and duplicates

### 2. Collaborative Filtering
- **User-based**: Finds similar users, recommends their favorites
- **Item-based**: Recommends movies similar to ones you've liked
- **Matrix Factorization**: Uses SVD to discover latent features
- Handles cold start problem with content-based fallback

### 3. Content-Based Filtering
- Analyzes movie features: genres, cast, crew, keywords, plot
- Creates similarity matrix using cosine similarity
- Recommends movies with similar content characteristics
- Works well for new movies with limited user data

### 4. GPT-4 Integration
- Understands natural language preferences
- Provides human-like explanations for recommendations
- Adds semantic understanding beyond traditional ML
- Generates contextual movie suggestions

### 5. Hybrid Scoring
- Combines all approaches using weighted scoring
- Boosts movies recommended by multiple methods
- Balances exploration vs. exploitation
- Provides confidence scores for recommendations

## 📊 Performance Features

- **Caching**: Streamlit caching for fast repeated queries
- **Optimization**: Efficient similarity computations
- **Scalability**: Handles large movie databases
- **Fallbacks**: Graceful degradation when APIs are unavailable

## 🔧 Configuration

### Recommendation Weights
Adjust in `hybrid_engine.py`:

```python
weights = {
    'collaborative': 0.4,  # User behavior patterns
    'content': 0.4,        # Movie content similarity  
    'gpt': 0.2             # AI-generated insights
}
```

### System Parameters
- **Max recommendations**: 1-20 movies
- **Similarity threshold**: Minimum score for recommendations
- **API timeout**: OpenAI request timeout
- **Cache duration**: How long to cache results

## 🧪 Testing

Run comprehensive tests:

```bash
python setup_and_run.py
# Choose option 2 for testing
```

Tests include:
- Data loading and preprocessing
- Content-based recommendations
- Collaborative filtering (if user data available)
- GPT integration (if API key provided)
- Trending movie algorithms

## 📈 Extending the System

### Adding New Data Sources
1. Extend `DataPreprocessor` class
2. Add new feature extraction methods
3. Update similarity calculations

### Custom Recommendation Algorithms
1. Create new recommender class
2. Implement recommendation interface
3. Add to hybrid scoring system

### Enhanced GPT Prompts
1. Modify prompt templates in `gpt_recommender.py`
2. Add context-specific prompting
3. Implement few-shot learning examples

## 🐛 Troubleshooting

**Common Issues:**

1. **"Data files not found"**
   - Download movie datasets to `data/` folder
   - Check file names match exactly

2. **"OpenAI API error"**
   - Verify API key in `.env` file
   - Check OpenAI account credits
   - System works without GPT features

3. **"Memory error during training"**
   - Reduce dataset size for testing
   - Use more powerful machine
   - Implement batch processing

4. **"Streamlit won't start"**
   - Check port 8501 is available
   - Try: `streamlit run streamlit_app.py --server.port 8502`

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 🙏 Acknowledgments

- **TMDB** for movie metadata API
- **GroupLens** for MovieLens datasets
- **OpenAI** for GPT-4 API
- **Streamlit** for the amazing web framework
- **scikit-learn** community for ML algorithms

## 📞 Support

- 📧 Email: your-email@example.com
- 🐛 Issues: GitHub Issues page
- 💬 Discussions: GitHub Discussions

---

**Built with ❤️ by [Your Name]**

*Discover your next favorite movie with the power of AI and machine learning!* 🎬✨