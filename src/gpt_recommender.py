# src/gpt_recommender.py
import openai
import os
import json
import pandas as pd
from typing import List, Dict, Optional
import time

class GPTRecommender:
    def __init__(self, api_key: str, movies_df: pd.DataFrame):
        self.client = openai.OpenAI(api_key=api_key)
        self.movies_df = movies_df
        
    def create_movie_context(self, movie_list: List[str], max_movies: int = 50) -> str:
        """Create context string with movie information for GPT"""
        context_movies = []
        
        for movie_title in movie_list[:max_movies]:
            movie_info = self.movies_df[
                self.movies_df['title'].str.lower() == movie_title.lower()
            ]
            
            if not movie_info.empty:
                movie = movie_info.iloc[0]
                context_movies.append({
                    'title': movie['title'],
                    'genres': movie.get('genres', ''),
                    'overview': movie.get('overview', '')[:200],  # Truncate for context
                    'release_year': str(movie.get('release_date', ''))[:4] if movie.get('release_date') else 'Unknown',
                    'rating': movie.get('vote_average', 0)
                })
                
        return json.dumps(context_movies, indent=2)
    
    def generate_recommendations_by_preference(
        self, 
        user_preferences: str, 
        n_recommendations: int = 5,
        excluded_movies: List[str] = None
    ) -> List[Dict]:
        """Generate movie recommendations based on user preferences using GPT"""
        
        excluded_movies = excluded_movies or []
        excluded_str = f"Do not recommend these movies: {', '.join(excluded_movies)}" if excluded_movies else ""
        
        # Create a sample of movies for context
        sample_movies = self.movies_df.sample(n=min(100, len(self.movies_df)))['title'].tolist()
        movie_context = self.create_movie_context(sample_movies)
        
        prompt = f"""
You are a movie recommendation expert. Based on the user's preferences, recommend {n_recommendations} movies from the available movie database.

User Preferences: {user_preferences}

{excluded_str}

Available Movies Context (sample):
{movie_context}

Please provide your recommendations in the following JSON format:
{{
  "recommendations": [
    {{
      "title": "Movie Title",
      "reason": "Brief explanation (2-3 sentences) of why this movie matches the user's preferences",
      "confidence": 0.85
    }}
  ],
  "explanation": "Overall explanation of your recommendation strategy"
}}

Focus on movies that truly match the user's described preferences. Provide clear, personalized reasoning for each recommendation.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a knowledgeable movie recommendation assistant. Provide thoughtful, personalized movie suggestions with clear reasoning."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Parse the response
            content = response.choices[0].message.content
            
            # Try to extract JSON from the response
            try:
                # Find JSON block in the response
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                json_str = content[start_idx:end_idx]
                
                result = json.loads(json_str)
                
                # Enhance recommendations with movie data from our database
                enhanced_recommendations = []
                for rec in result.get('recommendations', []):
                    movie_info = self.get_movie_details(rec['title'])
                    if movie_info:
                        enhanced_rec = {
                            **movie_info,
                            'gpt_reason': rec.get('reason', ''),
                            'gpt_confidence': rec.get('confidence', 0.5),
                            'recommendation_source': 'GPT-4'
                        }
                        enhanced_recommendations.append(enhanced_rec)
                
                return {
                    'recommendations': enhanced_recommendations,
                    'explanation': result.get('explanation', ''),
                    'success': True
                }
                
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    'recommendations': [],
                    'explanation': content,
                    'success': False,
                    'error': 'Failed to parse JSON response'
                }
                
        except Exception as e:
            print(f"Error generating GPT recommendations: {str(e)}")
            return {
                'recommendations': [],
                'explanation': f"Error: {str(e)}",
                'success': False,
                'error': str(e)
            }
    
    def generate_similar_movie_explanation(
        self, 
        base_movie: str, 
        recommended_movies: List[str]
    ) -> str:
        """Generate explanation for why recommended movies are similar to base movie"""
        
        base_movie_info = self.get_movie_details(base_movie)
        if not base_movie_info:
            return f"Could not find information about '{base_movie}'"
        
        recommended_info = [self.get_movie_details(movie) for movie in recommended_movies]
        recommended_info = [info for info in recommended_info if info]  # Filter out None values
        
        prompt = f"""
You are explaining movie recommendations. A user liked "{base_movie}" and you're recommending similar movies.

Base Movie Information:
- Title: {base_movie_info['title']}
- Genres: {base_movie_info.get('genres', 'Unknown')}
- Overview: {base_movie_info.get('overview', 'No overview available')}

Recommended Similar Movies:
"""
        
        for i, movie in enumerate(recommended_info[:3]):  # Limit to top 3 for context
            prompt += f"""
{i+1}. {movie['title']}
   - Genres: {movie.get('genres', 'Unknown')}
   - Overview: {movie.get('overview', 'No overview available')[:150]}...
"""
        
        prompt += f"""
Please provide a brief explanation (2-3 sentences) for each recommended movie, explaining why it's similar to "{base_movie}" and why someone who liked "{base_movie}" would enjoy it.

Format your response as:
"If you liked {base_movie}, here's why you'll love these recommendations:

1. [Movie Title]: [Explanation]
2. [Movie Title]: [Explanation]
3. [Movie Title]: [Explanation]"
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a movie expert who provides clear, engaging explanations for movie recommendations."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating explanation: {str(e)}")
            return f"Unable to generate explanation due to error: {str(e)}"
    
    def get_movie_details(self, movie_title: str) -> Optional[Dict]:
        """Get movie details from the database"""
        try:
            movie_info = self.movies_df[
                self.movies_df['title'].str.lower() == movie_title.lower()
            ]
            
            if movie_info.empty:
                # Try fuzzy matching
                matches = self.movies_df[
                    self.movies_df['title'].str.lower().str.contains(
                        movie_title.lower(), na=False
                    )
                ]
                if not matches.empty:
                    movie_info = matches.iloc[0:1]
                else:
                    return None
            
            movie = movie_info.iloc[0]
            
            return {
                'movie_id': movie.get('id'),
                'title': movie['title'],
                'overview': movie.get('overview', ''),
                'genres': movie.get('genres', ''),
                'release_date': movie.get('release_date', ''),
                'vote_average': movie.get('vote_average', 0),
                'vote_count': movie.get('vote_count', 0),
                'runtime': movie.get('runtime', 0)
            }
            
        except Exception as e:
            print(f"Error getting movie details for '{movie_title}': {str(e)}")
            return None
    
    def enhance_recommendations_with_gpt(
        self, 
        recommendations: List[Dict], 
        user_context: str = ""
    ) -> List[Dict]:
        """Enhance existing recommendations with GPT explanations"""
        
        movie_titles = [rec['title'] for rec in recommendations[:5]]  # Limit for API efficiency
        
        prompt = f"""
You are providing personalized explanations for movie recommendations.

{f"User Context: {user_context}" if user_context else ""}

Movies to explain:
{chr(10).join([f"- {title}" for title in movie_titles])}

For each movie, provide a compelling reason (1-2 sentences) why this would be a great recommendation, focusing on what makes each movie special and appealing.

Format as:
1. [Movie Title]: [Compelling reason]
2. [Movie Title]: [Compelling reason]
etc.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an enthusiastic movie expert who provides engaging, personalized explanations."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=400
            )
            
            explanations = response.choices[0].message.content.strip()
            
            # Parse explanations and add to recommendations
            explanation_lines = explanations.split('\n')
            
            enhanced_recs = []
            for i, rec in enumerate(recommendations):
                enhanced_rec = rec.copy()
                
                # Find matching explanation
                for line in explanation_lines:
                    if rec['title'].lower() in line.lower():
                        # Extract explanation after the colon
                        if ':' in line:
                            enhanced_rec['gpt_explanation'] = line.split(':', 1)[1].strip()
                            break
                
                if 'gpt_explanation' not in enhanced_rec:
                    enhanced_rec['gpt_explanation'] = f"A great movie that matches your preferences!"
                    
                enhanced_recs.append(enhanced_rec)
                
            return enhanced_recs
            
        except Exception as e:
            print(f"Error enhancing recommendations: {str(e)}")
            # Return original recommendations if enhancement fails
            for rec in recommendations:
                rec['gpt_explanation'] = "A recommended movie based on your preferences."
            return recommendations

if __name__ == "__main__":
    print("This is the GPT recommender module. Import and use in your main application.")