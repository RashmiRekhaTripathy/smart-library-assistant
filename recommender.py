import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import pickle
import os
import sqlite3

class BookRecommender:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.books_df = None
        self.knn_model = None
    
    def load_from_database(self, db_path='data/database.db'):
        """Load books from database"""
        print("📚 Loading books from database...")
        
        if not os.path.exists(db_path):
            print(f"❌ Database file {db_path} not found!")
            return None
        
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query('SELECT * FROM books', conn)
        conn.close()
        
        if len(df) == 0:
            print("⚠️ No books found in database.")
            return None
        
        self.books_df = df
        print(f"✅ Loaded {len(df)} books from database")
        return df
    
    def train(self):
        """Train recommendation models"""
        print("🤖 Training recommendation system...")
        
        if self.books_df is None:
            print("❌ No data to train. Load data first.")
            return self
        
        # Check if features column exists
        if 'features' not in self.books_df.columns:
            print("⚠️ Creating features column...")
            self.books_df['features'] = self.books_df['title'] + ' ' + self.books_df['author'] + ' ' + self.books_df['category']
            self.books_df['features'] = self.books_df['features'].fillna('')
        
        # TF-IDF Vectorization
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Fill NaN values
        features = self.books_df['features'].fillna('')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(features)
        
        # Train KNN
        n_neighbors = min(10, len(self.books_df))
        self.knn_model = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric='cosine',
            algorithm='brute'
        )
        self.knn_model.fit(self.tfidf_matrix)
        
        print(f"✅ Model trained on {len(self.books_df)} books")
        return self
    
    def search_books(self, query, top_n=10):
        """Search books using natural language query"""
        if self.tfidf_vectorizer is None or self.books_df is None:
            print("⚠️ Model not trained or no data")
            return []
        
        try:
            query_vec = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:top_n]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.05:  # Lower threshold for better results
                    book = self.books_df.iloc[idx]
                    results.append({
                        'title': book['title'],
                        'author': book['author'],
                        'category': book.get('category', 'Unknown'),
                        'rating': float(book.get('rating', 0)),
                        'price': float(book.get('price', 0)),
                        'score': float(similarities[idx])
                    })
            return results
        except Exception as e:
            print(f"⚠️ Error in search: {e}")
            return []
    
    def get_recommendations(self, book_title, n_recommendations=5):
        """Get book recommendations based on a book"""
        if self.knn_model is None or self.books_df is None:
            print("⚠️ Model not trained or no data")
            return []
        
        try:
            # Find book
            book_indices = self.books_df[self.books_df['title'].str.contains(book_title, case=False)].index
            if len(book_indices) == 0:
                print(f"❌ Book '{book_title}' not found")
                return []
            
            idx = book_indices[0]
            
            # Get nearest neighbors
            n_neighbors = min(n_recommendations + 1, len(self.books_df))
            distances, indices = self.knn_model.kneighbors(
                self.tfidf_matrix[idx],
                n_neighbors=n_neighbors
            )
            
            recommendations = []
            for i in range(1, len(indices[0])):
                book_idx = indices[0][i]
                book = self.books_df.iloc[book_idx]
                recommendations.append({
                    'title': book['title'],
                    'author': book['author'],
                    'category': book.get('category', 'Unknown'),
                    'rating': float(book.get('rating', 0)),
                    'price': float(book.get('price', 0)),
                    'score': float(1 - distances[0][i])
                })
            
            return recommendations
        except Exception as e:
            print(f"⚠️ Error getting recommendations: {e}")
            return []

def train_model():
    """Train and save the model"""
    print("=" * 50)
    print("Starting Model Training Process...")
    print("=" * 50)
    
    recommender = BookRecommender()
    
    # Load data from database
    df = recommender.load_from_database()
    
    if df is None or len(df) == 0:
        print("\n⚠️ No data in database. Loading from CSV...")
        try:
            from data_loader import load_data, preprocess_data, save_to_database
            df_raw = load_data()
            if df_raw is not None:
                df_clean = preprocess_data(df_raw)
                save_to_database(df_clean)
                recommender.load_from_database()
            else:
                print("❌ Failed to load data from CSV")
                return None
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    # Train model
    if recommender.books_df is not None and len(recommender.books_df) > 0:
        recommender.train()
        
        # Save model
        os.makedirs('models', exist_ok=True)
        with open('models/recommender.pkl', 'wb') as f:
            pickle.dump(recommender, f)
        
        print("\n" + "=" * 50)
        print("✅ Model saved to models/recommender.pkl")
        print("=" * 50)
        return recommender
    else:
        print("❌ No data available for training")
        return None

def load_model():
    """Load trained recommendation model"""
    try:
        model_path = 'models/recommender.pkl'
        if not os.path.exists(model_path):
            print("⚠️ No model found. Training new model...")
            return train_model()
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            print("✅ Model loaded successfully")
            return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Training new model...")
        return train_model()

if __name__ == "__main__":
    train_model()