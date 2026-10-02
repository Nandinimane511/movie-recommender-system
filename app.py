import pickle
import streamlit as st
import requests
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
import random
import pandas as pd

# Page setup
st.set_page_config(
    page_title="CineMatch - AI Movie Recommender",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Responsive Cinema Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 4px;
        text-align: center;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        text-align: center;
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 25px;
        font-weight: 400;
    }
    
    .movie-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 12px;
        margin-bottom: 20px;
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.4);
    }
    
    .movie-card:hover {
        transform: translateY(-6px);
        border-color: #a855f7;
        box-shadow: 0 14px 28px rgba(168, 85, 247, 0.25);
    }
    
    .poster-img {
        width: 100%;
        aspect-ratio: 2 / 3;
        max-height: 310px;
        object-fit: cover;
        border-radius: 10px;
        margin-bottom: 8px;
        background-color: #1e293b;
    }
    
    .movie-title {
        font-weight: 700;
        font-size: 0.95rem;
        color: #f8fafc;
        margin: 4px 0 6px 0;
        line-height: 1.3;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        min-height: 2.5rem;
    }
    
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
        margin-bottom: 6px;
    }
    
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 2px 7px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        white-space: nowrap;
    }
    
    .badge-similarity {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    
    .badge-rating {
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    
    .badge-genre {
        background: rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.4);
    }
    
    .meta-text {
        font-size: 0.76rem;
        color: #94a3b8;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        padding-top: 6px;
        margin-top: 4px;
    }

    .stButton>button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #d946ef);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 0.95rem;
        transition: all 0.25s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(168, 85, 247, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_data():
    movies = pickle.load(open('model/movie_list.pkl', 'rb'))
    similarity = pickle.load(open('model/similarity.pkl', 'rb'))
    return movies, similarity

movies, similarity = load_data()

# Fast Multi-source Poster Fetcher
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_poster(title):
    # Try OMDb API
    try:
        clean_title = title.split('(')[0].strip()
        omdb_url = f"http://www.omdbapi.com/?t={urllib.parse.quote(clean_title)}&apikey=trilogy"
        res = requests.get(omdb_url, timeout=1.8)
        if res.status_code == 200:
            poster = res.json().get('Poster')
            if poster and poster.startswith('http'):
                return poster
    except Exception:
        pass

    # Try Wikipedia Page Summary
    try:
        headers = {'User-Agent': 'CineMatch/2.0 (student-ml-app)'}
        s_url = f"https://en.wikipedia.org/w/rest.php/v1/search/page?q={urllib.parse.quote(title)}+film&limit=1"
        r = requests.get(s_url, headers=headers, timeout=1.8).json()
        pages = r.get('pages', [])
        if pages:
            p_key = pages[0]['key']
            sum_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{p_key}"
            sum_res = requests.get(sum_url, headers=headers, timeout=1.8).json()
            thumb = sum_res.get('thumbnail', {}).get('source')
            if thumb and thumb.startswith('http'):
                return thumb
    except Exception:
        pass

    # Fallback high quality poster image
    return "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&auto=format&fit=crop&q=80"


def get_recommendations(movie_title, top_k=5, selected_genre="All"):
    try:
        idx = movies[movies['title'] == movie_title].index[0]
    except IndexError:
        return []

    distances = list(enumerate(similarity[idx]))
    sorted_dists = sorted(distances, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i, score in sorted_dists[1:]:
        movie_row = movies.iloc[i]
        genres = movie_row.get('genres_list', [])
        
        if selected_genre != "All" and selected_genre not in genres:
            continue
            
        rec = {
            'index': i,
            'title': movie_row['title'],
            'score': round(float(score) * 100, 1),
            'rating': movie_row.get('vote_average', 0.0),
            'genres': genres[:2],
            'director': movie_row.get('director', 'Unknown'),
            'cast': movie_row.get('cast_list', [])[:2],
            'year': str(movie_row.get('release_date', ''))[:4] if movie_row.get('release_date') else 'N/A',
            'runtime': f"{movie_row.get('runtime', 0)} min" if movie_row.get('runtime') else 'N/A',
            'overview': movie_row.get('overview', 'No synopsis available.')
        }
        recommendations.append(rec)
        if len(recommendations) >= top_k:
            break

    # Parallel poster fetching
    with ThreadPoolExecutor(max_workers=min(top_k, 8)) as executor:
        titles = [r['title'] for r in recommendations]
        posters = list(executor.map(fetch_poster, titles))

    for rec, poster in zip(recommendations, posters):
        rec['poster'] = poster

    return recommendations


def render_movie_grid(recs, cols_per_row=4):
    """Renders movies into a clean, responsive multi-row grid."""
    num_recs = len(recs)
    
    # Dynamically choose optimal column count
    if num_recs <= 3:
        actual_cols_per_row = num_recs
    elif num_recs == 5 or num_recs == 10:
        actual_cols_per_row = 5
    else:
        actual_cols_per_row = 4

    for chunk_start in range(0, num_recs, actual_cols_per_row):
        chunk = recs[chunk_start:chunk_start + actual_cols_per_row]
        cols = st.columns(actual_cols_per_row)
        
        for col_idx, rec in enumerate(chunk):
            with cols[col_idx]:
                genres_html = "".join([f'<span class="badge badge-genre">{g}</span>' for g in rec['genres']])
                
                st.markdown(f"""
                <div class="movie-card">
                    <img src="{rec['poster']}" class="poster-img" onerror="this.onerror=null;this.src='https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&auto=format&fit=crop&q=80';">
                    <div class="movie-title" title="{rec['title']}">{rec['title']}</div>
                    <div class="badge-container">
                        <span class="badge badge-similarity">⚡ {rec['score']}%</span>
                        <span class="badge badge-rating">⭐ {rec['rating']}</span>
                    </div>
                    <div class="badge-container">{genres_html}</div>
                    <div class="meta-text">📅 {rec['year']} &nbsp;|&nbsp; ⏱️ {rec['runtime']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("📖 Plot & Details"):
                    st.caption(f"**🎬 Director:** {rec['director']}")
                    st.caption(f"**🎭 Cast:** {', '.join(rec['cast'])}")
                    st.caption(f"**📝 Synopsis:** {rec['overview'][:180]}...")


# --- Sidebar ---
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1536440136628-849c177e76a1?w=600&auto=format&fit=crop&q=80")
    st.markdown("### ⚙️ Recommendation Settings")
    
    top_n = st.slider("Number of Recommendations:", min_value=3, max_value=12, value=8, step=1)
    
    all_genres = ["All", "Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", 
                  "Family", "Fantasy", "Horror", "Mystery", "Romance", "Science Fiction", "Thriller"]
    genre_filter = st.selectbox("🎯 Filter by Genre:", all_genres)
    
    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.info(f"""
    * **Total Movies**: {len(movies):,}
    * **ML Model**: Cosine Similarity
    * **Features**: Genres, Cast, Crew, Overview
    * **NLP**: CountVectorizer (5000 feats)
    """)


# --- Header ---
st.markdown('<div class="main-header">🍿 CineMatch AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Content-Based Movie Recommendation Engine Powered by Machine Learning & NLP</div>', unsafe_allow_html=True)

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🎯 Movie Recommender", "🎲 Surprise Me", "📈 Model & Analytics"])

with tab1:
    col_search, col_btn = st.columns([4, 1])
    
    with col_search:
        selected_movie = st.selectbox(
            "Select or Type a Movie You Enjoy:",
            movies['title'].values,
            index=0,
            key="movie_select"
        )
    
    with col_btn:
        st.write("") 
        st.write("") 
        recommend_btn = st.button("🚀 Recommend")

    # Selected Movie Details Banner
    sel_movie_data = movies[movies['title'] == selected_movie].iloc[0]
    with st.expander(f"ℹ️ Selected Movie: **{selected_movie}** ({str(sel_movie_data.get('release_date', ''))[:4]})", expanded=False):
        c1, c2, c3 = st.columns([1, 1, 1])
        c1.metric("⭐ TMDB Rating", f"{sel_movie_data.get('vote_average', 0)} / 10")
        c2.metric("⏱️ Runtime", f"{sel_movie_data.get('runtime', 0)} mins")
        c3.metric("🎬 Director", f"{sel_movie_data.get('director', 'Unknown')}")
        st.markdown(f"**Genres:** {', '.join(sel_movie_data.get('genres_list', []))}")
        st.markdown(f"**Top Cast:** {', '.join(sel_movie_data.get('cast_list', []))}")
        st.markdown(f"**Synopsis:** {sel_movie_data.get('overview', 'N/A')}")

    # Recommendations Output
    if recommend_btn or 'has_run' in st.session_state:
        st.session_state['has_run'] = True
        
        with st.spinner("✨ Finding matches and loading posters..."):
            recs = get_recommendations(selected_movie, top_k=top_n, selected_genre=genre_filter)

        if not recs:
            st.warning(f"No recommendations found matching the genre '{genre_filter}'. Try selecting 'All' genres.")
        else:
            st.markdown(f"### 🎯 Top {len(recs)} Recommendations for *{selected_movie}*:")
            render_movie_grid(recs)


with tab2:
    st.markdown("### 🎲 Feeling Lucky? Let AI Pick a Surprise Movie!")
    if st.button("🎰 Pick a Random Movie & Recommend", key="random_btn"):
        random_movie = random.choice(movies['title'].values)
        st.success(f"🎉 Selected: **{random_movie}**")
        
        with st.spinner("Finding matches..."):
            random_recs = get_recommendations(random_movie, top_k=top_n, selected_genre=genre_filter)
            if random_recs:
                render_movie_grid(random_recs)


with tab3:
    st.markdown("### 📐 Machine Learning Pipeline & Analytics")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Movies Indexed", f"{len(movies):,}")
    c2.metric("Feature Vector Size", "5,000 Dimensions")
    c3.metric("Similarity Metric", "Cosine Distance (NLP)")

    st.markdown("#### 🔍 How the Recommendation Works (ML Workflow)")
    st.markdown("""
    1. **Data Ingestion & Cleaning**: Merged TMDB 5000 movies and credits datasets, extracting director, top 3 cast members, genres, keywords, and plot overview.
    2. **NLP Preprocessing**: Tokenized strings, converted them to lowercase, removed entity spaces, and stemmed words using **NLTK PorterStemmer** to normalize variations.
    3. **Vectorization**: Transformed metadata tags into a 5,000-dimensional Bag-of-Words matrix using `CountVectorizer(stop_words='english')`.
    4. **Cosine Similarity**: Measured high-dimensional cosine angle between movie vectors:
    """)
    st.latex(r"\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}")
    
    st.markdown("#### 📊 Top Movie Genres in Dataset")
    all_g_flat = [g for sublist in movies['genres_list'] for g in sublist]
    genre_counts = pd.Series(all_g_flat).value_counts().head(10)
    st.bar_chart(genre_counts)
