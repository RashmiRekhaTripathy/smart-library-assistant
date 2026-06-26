import pandas as pd
import sqlite3
import os
import random

def load_data(data_path='data/raw/'):
    """Load the Amazon bestselling books dataset"""
    print("📚 Loading Amazon Bestselling Books dataset...")
    
    # Find the CSV file
    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    
    if not csv_files:
        print("⚠️ No CSV files found in data/raw/")
        print("Please place amazon_bestselling_books_500.csv in data/raw/")
        return None
    
    # Load the dataset
    data_file = os.path.join(data_path, csv_files[0])
    df = pd.read_csv(data_file)
    
    print(f"✅ Loaded {len(df)} books")
    print(f"\n📊 Columns: {df.columns.tolist()}")
    print(f"\n📋 First 5 rows:")
    print(df.head())
    
    return df

def preprocess_data(df):
    """Clean and preprocess the data"""
    print("\n🧹 Cleaning and preprocessing data...")
    
    df_clean = df.copy()
    
    # Rename columns for our database
    # Your CSV uses: Title, Author, Category, Rating, Year Published, Price (USD)
    df_clean.rename(columns={
        'Title': 'title',
        'Author': 'author',
        'Category': 'category',
        'Sub-Genre': 'sub_genre',
        'Rating': 'rating',
        'Reviews': 'num_reviews',
        'Year Published': 'year',
        'Price (USD)': 'price',
        'Format': 'format',
        'Publisher': 'publisher',
        'Rank': 'rank',
        'Weeks on List': 'weeks_on_list',
        'ISBN': 'isbn',
        'Amazon BSR': 'amazon_bsr'
    }, inplace=True)
    
    # Clean text columns
    df_clean['title'] = df_clean['title'].astype(str).str.strip()
    df_clean['author'] = df_clean['author'].astype(str).str.strip()
    df_clean['category'] = df_clean['category'].astype(str).str.strip()
    
    # Create features for recommendation
    df_clean['features'] = df_clean['title'] + ' ' + df_clean['author'] + ' ' + df_clean['category']
    df_clean['features'] = df_clean['features'].fillna('')
    
    # Add copies column (random for demo)
    random.seed(42)
    df_clean['copies'] = [random.randint(1, 5) for _ in range(len(df_clean))]
    
    # Handle missing values
    df_clean['rating'] = pd.to_numeric(df_clean['rating'], errors='coerce').fillna(0)
    df_clean['price'] = pd.to_numeric(df_clean['price'], errors='coerce').fillna(0)
    df_clean['year'] = pd.to_numeric(df_clean['year'], errors='coerce').fillna(2000).astype(int)
    
    print(f"✅ Data cleaned: {len(df_clean)} records")
    print(f"📊 Columns after cleaning: {df_clean.columns.tolist()}")
    return df_clean

def save_to_database(df, db_path='data/database.db'):
    """Save to SQLite database"""
    print("\n💾 Saving to database...")
    
    conn = sqlite3.connect(db_path)
    
    # Create table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS books (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            author TEXT,
            category TEXT,
            sub_genre TEXT,
            year INTEGER,
            rating REAL,
            price REAL,
            copies INTEGER DEFAULT 1,
            features TEXT,
            publisher TEXT,
            format TEXT,
            rank INTEGER,
            weeks_on_list INTEGER,
            isbn TEXT,
            amazon_bsr INTEGER
        )
    ''')
    
    # Clear existing data
    conn.execute('DELETE FROM books')
    
    # Insert data
    for _, row in df.iterrows():
        conn.execute('''
            INSERT INTO books (
                title, author, category, sub_genre, year, rating, price, 
                copies, features, publisher, format, rank, weeks_on_list, isbn, amazon_bsr
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            row['title'], 
            row['author'], 
            row['category'],
            row.get('sub_genre', ''), 
            row['year'], 
            row['rating'], 
            row['price'],
            row['copies'], 
            row['features'],
            row.get('publisher', ''), 
            row.get('format', ''),
            row.get('rank', 0),
            row.get('weeks_on_list', 0),
            row.get('isbn', ''),
            row.get('amazon_bsr', 0)
        ))
    
    conn.commit()
    count = conn.execute('SELECT COUNT(*) FROM books').fetchone()[0]
    print(f"✅ {count} books inserted into database")
    conn.close()
    return count

if __name__ == "__main__":
    print("=" * 50)
    print("Starting Data Loading Process...")
    print("=" * 50)
    
    df = load_data()
    if df is not None:
        print(f"\n📊 Dataset Info:")
        print(f"   - Shape: {df.shape}")
        print(f"   - Columns: {df.columns.tolist()}")
        
        df_clean = preprocess_data(df)
        save_to_database(df_clean)
        print("\n" + "=" * 50)
        print("🎉 Data loading complete!")
        print("=" * 50)
    else:
        print("❌ Data loading failed!")