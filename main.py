import streamlit as st
import sys
import os
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.recommender import load_model
except ImportError as e:
    st.error(f"Error importing recommender: {e}")
    st.stop()

# Page config
st.set_page_config(
    page_title="Smart Library Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f3a93;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .book-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.7rem 0;
        border-left: 5px solid #3498db;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .book-card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .book-title {
        font-size: 1.1rem;
        font-weight: bold;
        color: #1f3a93;
    }
    .book-author {
        color: #555;
        font-size: 0.95rem;
    }
    .book-meta {
        color: #666;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }
    .rating-badge {
        background: #f39c12;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        font-size: 0.8rem;
        display: inline-block;
    }
    .price-badge {
        background: #27ae60;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        font-size: 0.8rem;
        display: inline-block;
        margin-left: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'recommender' not in st.session_state:
    with st.spinner("🔄 Loading AI Model... This may take a few seconds."):
        try:
            st.session_state.recommender = load_model()
            if st.session_state.recommender is None:
                st.error("❌ Failed to load recommendation model. Please check your data.")
                st.stop()
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            st.stop()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/book.png", width=80)
    st.title("📚 Library Assistant")
    st.markdown("---")
    
    menu = st.radio(
        "📌 Navigate",
        ["🏠 Home", "🔍 Search Books", "📖 Recommendations", "📊 About"]
    )
    
    st.markdown("---")
    
    # Show statistics in sidebar
    if st.session_state.recommender and st.session_state.recommender.books_df is not None:
        books = st.session_state.recommender.books_df
        st.metric("📖 Total Books", len(books))
        
        if 'category' in books.columns:
            categories = books['category'].value_counts()
            if len(categories) > 0:
                st.metric("📚 Categories", len(categories))

# Main content
if menu == "🏠 Home":
    st.markdown('<div class="main-header">📚 Smart Library Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Book Discovery & Recommendations</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    if st.session_state.recommender and st.session_state.recommender.books_df is not None:
        books = st.session_state.recommender.books_df
        
        with col1:
            st.metric("📖 Total Books", len(books))
        with col2:
            avg_rating = books['rating'].mean() if 'rating' in books.columns else 0
            st.metric("⭐ Avg Rating", f"{avg_rating:.1f}")
        with col3:
            categories = books['category'].nunique() if 'category' in books.columns else 0
            st.metric("📚 Categories", categories)
        with col4:
            authors = books['author'].nunique() if 'author' in books.columns else 0
            st.metric("✍️ Authors", authors)
    
    st.markdown("---")
    
    st.subheader("✨ Quick Start")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.info("🔍 **Search for Books**\n\nFind your favorite books by title or author.")
        if st.button("Go to Search", use_container_width=True):
            st.session_state.menu = "🔍 Search Books"
            st.rerun()
    
    with col_b:
        st.info("📖 **Get Recommendations**\n\nDiscover new books based on your favorites.")
        if st.button("Get Recommendations", use_container_width=True):
            st.session_state.menu = "📖 Recommendations"
            st.rerun()
    
    # Show popular books
    st.markdown("---")
    st.subheader("🔥 Popular Books")
    
    if st.session_state.recommender and st.session_state.recommender.books_df is not None:
        books = st.session_state.recommender.books_df
        if 'rating' in books.columns:
            popular = books.nlargest(5, 'rating')
            for _, book in popular.iterrows():
                st.markdown(f"""
                <div class="book-card">
                    <div class="book-title">{book['title']}</div>
                    <div class="book-author">✍️ {book['author']}</div>
                    <div class="book-meta">
                        <span class="rating-badge">⭐ {book['rating']:.1f}</span>
                        <span class="price-badge">💰 ${book['price']:.2f}</span>
                        <span style="margin-left: 0.5rem;">📂 {book.get('category', 'N/A')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

elif menu == "🔍 Search Books":
    st.header("🔍 Search Books")
    st.markdown("Search by title, author, or category")
    
    query = st.text_input(
        "Enter your search query:",
        placeholder="e.g., Harry Potter, James Clear, Fiction...",
        help="Search for books by title, author, or category"
    )
    
    if query:
        with st.spinner("🔍 Searching..."):
            results = st.session_state.recommender.search_books(query)
            
            if results:
                st.success(f"✅ Found {len(results)} results for '{query}'")
                for book in results:
                    st.markdown(f"""
                    <div class="book-card">
                        <div class="book-title">{book['title']}</div>
                        <div class="book-author">✍️ {book['author']}</div>
                        <div class="book-meta">
                            <span class="rating-badge">⭐ {book['rating']:.1f}</span>
                            <span class="price-badge">💰 ${book['price']:.2f}</span>
                            <span style="margin-left: 0.5rem;">📂 {book.get('category', 'N/A')}</span>
                            <span style="margin-left: 0.5rem;">🎯 Match: {book['score']:.2%}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info(f"😕 No books found for '{query}'. Try a different search term.")

elif menu == "📖 Recommendations":
    st.header("📖 Book Recommendations")
    st.markdown("Select a book you like and get personalized recommendations")
    
    if st.session_state.recommender and st.session_state.recommender.books_df is not None:
        books = st.session_state.recommender.books_df
        
        if len(books) > 0:
            # Get book titles for selection
            book_titles = books['title'].tolist()
            
            selected = st.selectbox(
                "Choose a book you enjoy:",
                book_titles,
                help="Select a book to get similar recommendations"
            )
            
            if st.button("🎯 Get Recommendations", use_container_width=True):
                with st.spinner("🔄 Finding similar books..."):
                    recs = st.session_state.recommender.get_recommendations(selected)
                    
                    if recs:
                        st.subheader("📚 You might also like:")
                        for book in recs:
                            st.markdown(f"""
                            <div class="book-card">
                                <div class="book-title">{book['title']}</div>
                                <div class="book-author">✍️ {book['author']}</div>
                                <div class="book-meta">
                                    <span class="rating-badge">⭐ {book['rating']:.1f}</span>
                                    <span class="price-badge">💰 ${book['price']:.2f}</span>
                                    <span style="margin-left: 0.5rem;">📂 {book.get('category', 'N/A')}</span>
                                    <span style="margin-left: 0.5rem;">🎯 Similarity: {book['score']:.2%}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("😕 No recommendations found for this book. Try another book.")
        else:
            st.warning("📚 No books available. Please check your data.")
    else:
        st.error("❌ No data loaded. Please check your database connection.")

elif menu == "📊 About":
    st.header("📊 About Smart Library Assistant")
    
    st.markdown("""
    ### 🤖 AI-Powered Library Management System
    
    **Smart Library Assistant** is an intelligent library management system that enables:
    
    - 🔍 **Natural Language Search**: Find books using everyday language
    - 📖 **Personalized Recommendations**: Get book suggestions based on your preferences
    - 📊 **Data-Driven Insights**: Powered by machine learning algorithms
    
    ### 🛠️ Technology Stack
    
    - **Frontend**: Streamlit
    - **Backend**: Python 3.10+
    - **Machine Learning**: scikit-learn, Pandas, NumPy
    - **Database**: SQLite
    - **AI Features**: TF-IDF Vectorization, Cosine Similarity, KNN
    
    ### 📊 Dataset
    
    This project uses the **Amazon Bestselling Books** dataset from Kaggle, containing:
    - 500+ bestselling books
    - Author, category, rating, price, and more
    - Real-world book data for accurate recommendations
    
    ### 📈 Features
    
    - **500+ Books** in the database
    - **Search** by title, author, or category
    - **Smart Recommendations** based on content similarity
    - **Beautiful UI** with book cards and ratings
    """)
    
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit | Data from Amazon Bestselling Books")

# Footer
st.markdown("---")
st.caption("📚 Smart Library Assistant v1.0 | Powered by AI")