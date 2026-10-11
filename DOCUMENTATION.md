# 🎬 CineMatch AI: System Documentation & Architecture Guide

**Project Title:** CineMatch AI — Content-Based Movie Recommendation System  
**Author:** Nandini Mane  
**Domain:** Machine Learning, Natural Language Processing (NLP) & Full-Stack Web Development  
**GitHub Repository:** [Nandinimane511/movie-recommender-system](https://github.com/Nandinimane511/movie-recommender-system)  

---

## 📌 1. Project Overview & Objective

**CineMatch AI** is an intelligent, high-performance Content-Based Movie Recommendation System built on the **TMDB 5000 Movie Dataset**. 

Unlike collaborative filtering systems that suffer from the *"cold-start problem"* (inability to recommend items to new users with no history), CineMatch AI operates on deep content metadata: **plot summaries, genre classifications, thematic keywords, top-billed actors, and film directors**.

The system translates unstructured metadata into a **5,000-dimensional vector space**, trains an unsupervised **K-Nearest Neighbors (KNN)** Machine Learning model using **Cosine Distance**, and serializes model weights into disk-cached binary artifacts. A responsive **Streamlit** web application serves recommendations with sub-5ms query latency and live poster enrichment via external REST APIs.

---

## 🏗️ 2. High-Level Architectural Flow

The architecture follows a decoupled 4-tier design:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   TIER 1: OFFLINE TRAINING & NLP PIPELINE              │
│  tmdb_5000_movies.csv + tmdb_5000_credits.csv                          │
│     │                                                                  │
│     ▼ (Clean, Merge on Title, Reset Index)                             │
│  Unified DataFrame (4,806 Movies)                                      │
│     │                                                                  │
│     ▼ (AST JSON Parsing: Cast, Crew, Director, Genres, Keywords)       │
│  Entity Space Collapsing ("Sam Worthington" -> "SamWorthington")        │
│     │                                                                  │
│     ▼ (Porter Stemming via NLTK)                                       │
│  Unified Stemmed Tags String                                           │
│     │                                                                  │
│     ▼ (CountVectorizer max_features=5000)                              │
│  Bag-of-Words Feature Matrix X ∈ ℝ^(4806 × 5000)                       │
│     │                                                                  │
│     ├─────────────────────────────────┐                                │
│     ▼                                 ▼                                │
│  Pairwise Cosine Similarity       NearestNeighbors (KNN)               │
│  Matrix S ∈ [0, 1]^(4806 × 4806)  (metric='cosine', algorithm='brute') │
└───────────────────────┬─────────────────┬──────────────────────────────┘
                        │                 │
                        ▼ (Serialization) ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   TIER 2: SERIALIZATION BRIDGE (model/)                │
│  • model/movie_list.pkl  (Clean DataFrame with metadata)               │
│  • model/similarity.pkl  (Precomputed 4,806 × 4,806 Cosine Matrix)    │
│  • model/knn_model.pkl   (Fitted K-Nearest Neighbors ML Model)         │
└───────────────────────────────────────┬────────────────────────────────┘
                                        │
                                        ▼ (Loaded once into RAM)
┌────────────────────────────────────────────────────────────────────────┐
│                   TIER 3: BACKEND ENGINE (app.py)                      │
│  • @st.cache_resource: In-memory RAM caching                           │
│  • Vector Index Lookup: S[idx] or knn_model.kneighbors() (<5ms)       │
│  • ThreadPoolExecutor: Parallel concurrent poster fetcher (OMDb / Wiki)│
└───────────────────────────────────────┬────────────────────────────────┘
                                        │
                                        ▼ (JSON Payloads & Image URLs)
┌────────────────────────────────────────────────────────────────────────┐
│                   TIER 4: FRONTEND PRESENTATION (app.py)               │
│  • Streamlit Web UI with Custom Cinema Glassmorphism CSS               │
│  • Responsive Movie Cards with Hover Elevation Animations              │
│  • Dynamic Badges (Match %, TMDB Star Rating, Genre Pills)             │
│  • Expandable Modals: Full Synopsis, Director, Cast, YouTube Trailers  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📓 3. The Jupyter Notebook (`Movie_Recommendation_System_EDA.ipynb`)

The primary development and experimentation environment is [Movie_Recommendation_System_EDA.ipynb](file:///d:/movie%20recommendation%20project/Movie_Recommendation_System_EDA.ipynb). It is fully executed and divided into 9 sequential sections:

| Section # | Notebook Section Title | Code / Process Performed | Outputs Embedded in Notebook |
|---|---|---|---|
| **Section 1** | Import Libraries & Load Datasets | Imports `pandas`, `numpy`, `sklearn`, `nltk`, `matplotlib`, `seaborn`. Loads TMDB CSVs. | Dataset dimensions: Movies $(4803, 20)$, Credits $(4803, 4)$. |
| **Section 2** | Exploratory Data Analysis: Correlation Matrix | Computes Pearson correlation across `budget`, `popularity`, `revenue`, `runtime`, `vote_average`, `vote_count`. | Formatted correlation matrix table + Annotated Seaborn **Blues** Heatmap (*Fig 2.1*). |
| **Section 3** | Dataset Merging & Feature Selection | Merges on `title`, drops nulls, retains essential attributes, and applies `reset_index(drop=True)`. | Cleaned DataFrame: exactly **4,806** movie records. |
| **Section 4** | Metadata Extraction | Uses `ast.literal_eval` to extract JSON arrays into Python lists: top 3 cast members, director, genres, and keywords. | Preview table displaying extracted list features. |
| **Section 5** | NLP Preprocessing & Unified Tag Construction | Collapses entity spaces (`Christopher Nolan` $\rightarrow$ `ChristopherNolan`), concatenates tokens into single tag string, and applies `PorterStemmer`. | Sample stemmed tag text string for movie index 0 (*Avatar*). |
| **Section 6** | Feature Matrix 1: Bag-of-Words Vector Space | Fits `CountVectorizer(max_features=5000, stop_words='english')`. | Matrix shape $(4806 \times 5000)$, Non-zero count: $145,410$, Sparsity: **$99.39\%$**, and sample slice DataFrame. |
| **Section 7** | Feature Matrix 2: Pairwise Cosine Similarity | Computes pairwise `cosine_similarity(vectors)`. Verifies diagonal identity ($1.00$). | Cosine matrix shape $(4806 \times 4806)$ + Annotated Submatrix Heatmap across benchmark movies (*Fig 7.1*). |
| **Section 8** | **Machine Learning Model: K-Nearest Neighbors (KNN)** | Trains `NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')`. Implements `recommend_with_knn()`. | Model architecture verification + Top-5 recommendation inference tests for *The Dark Knight*, *Interstellar*, *The Avengers*. |
| **Section 9** | Export Trained Model Artifacts | Uses Python `pickle` to export `movie_list.pkl`, `similarity.pkl`, and `knn_model.pkl` into the `model/` directory. | Confirmation message with total indexed movies count ($4,806$). |

---

## 🤖 4. Machine Learning Model: K-Nearest Neighbors (KNN)

### Why K-Nearest Neighbors?
In recommendation systems, KNN is a standard unsupervised metric-learning algorithm. It represents movies as coordinates in a 5,000-dimensional continuous space and indexes their neighborhood topology.

### Mathematical Formulation
Instead of Euclidean distance (which is sensitive to varying document lengths), KNN uses **Cosine Distance**:

$$\text{Cosine Distance } d(\vec{u}, \vec{v}) = 1.0 - \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|} = 1.0 - \frac{\sum_{i=1}^{D} u_i v_i}{\sqrt{\sum_{i=1}^{D} u_i^2} \sqrt{\sum_{i=1}^{D} v_i^2}}$$

Where:
- $\vec{u}, \vec{v} \in \mathbb{R}^{5000}$: Document-term frequency vectors.
- $D = 5,000$: Dimensionality of vocabulary space.
- $d(\vec{u}, \vec{v}) \in [0.0, 1.0]$: $0.0$ indicates identical tag distribution; $1.0$ indicates orthogonal/unrelated topics.

The **Similarity Match Percentage** shown to the user is derived directly as:
$$\text{Match Percentage} = (1.0 - d(\vec{u}, \vec{v})) \times 100\%$$

### Model Specifications
- **Class:** `sklearn.neighbors.NearestNeighbors`
- **Metric:** `cosine`
- **Algorithm:** `brute` (computes exact distances without tree approximation distortion)
- **Neighbors ($k$):** $6$ ($1$ query movie self-match $+$ top $5$ recommendations)

---

## 🎯 5. Step-by-Step Concrete Walkthrough Example

Consider a user requesting recommendations for **"The Dark Knight"**:

### Step 1: Raw Input Metadata
- **Title:** The Dark Knight (2008)
- **Director:** Christopher Nolan
- **Top 3 Cast:** Christian Bale, Heath Ledger, Aaron Eckhart
- **Genres:** Drama, Action, Crime, Thriller
- **Plot Overview:** *"Batman raises the stakes in his war on crime. With the help of Lt. Jim Gordon and District Attorney Harvey Dent, Batman sets out to dismantle the remaining criminal organizations that plague the streets..."*

### Step 2: NLP Tokenization & Stemming
The system parses and stems all entities:
- Entities collapsed: `Christian Bale` $\rightarrow$ `christianbale`, `Christopher Nolan` $\rightarrow$ `christophernolan`
- Stemmed Tag:  
  `"batman rais the stake in his war on crime with the help of lt jim gordon and district attorney harvey dent batman set out to dismantl the remain crimin organ that plagu the street ... christianbale heathledger aaroneckhart christophernolan"`

### Step 3: Vectorization & ML Model Query
1. Locate index: `idx = 65`.
2. Extract query vector: `query_vector = vectors[65].reshape(1, -1)`.
3. Execute KNN inference: `distances, indices = knn_model.kneighbors(query_vector, n_neighbors=6)`.

### Step 4: Top-5 Generated Recommendations

| Rank | Recommended Movie | Cosine Distance | Similarity Match % | TMDB Rating | Shared Content Attributes |
|:---:|---|:---:|:---:|:---:|---|
| **1** | **The Dark Knight Rises** | `0.5774` | **42.3%** | 7.6 / 10 | Same director (Nolan), lead actor (Bale), DC Gotham universe |
| **2** | **Batman Begins** | `0.5978` | **40.2%** | 7.5 / 10 | Origin story, same director (Nolan), lead actor (Bale) |
| **3** | **Batman Returns** | `0.6686` | **33.1%** | 6.6 / 10 | DC Comics, Gotham City vigilante, superhero crime thriller |
| **4** | **Batman Forever** | `0.7059` | **29.4%** | 5.2 / 10 | DC Comics, Gotham villains, Batmobile, masked vigilante |
| **5** | **Batman** | `0.7233` | **27.7%** | 7.0 / 10 | Classic Gotham lore, Joker adversary, Tim Burton DC adaptation |

---

## 🔌 6. Backend Integration (`app.py`)

The backend server is implemented in Python using Streamlit.

### 1. In-Memory Model Caching (`@st.cache_resource`)
Recomputing the similarity matrix on every request would take 20+ seconds. The backend loads the pre-trained weights once into RAM:

```python
@st.cache_resource
def load_data():
    movies = pickle.load(open('model/movie_list.pkl', 'rb'))
    similarity = pickle.load(open('model/similarity.pkl', 'rb'))
    knn_model = pickle.load(open('model/knn_model.pkl', 'rb'))
    return movies, similarity, knn_model

movies, similarity, knn_model = load_data()
```

### 2. Sub-5 Millisecond Inference
Given a query movie, recommendations are produced in less than 5 milliseconds:
```python
def get_recommendations(movie_title, top_k=5, selected_genre="All"):
    idx = movies[movies['title'] == movie_title].index[0]
    distances = list(enumerate(similarity[idx]))
    sorted_dists = sorted(distances, key=lambda x: x[1], reverse=True)
    # Filter by user genre and return top_k candidates
```

### 3. Concurrent Multi-Source Poster Pipeline
Raw TMDB CSV files do not contain image URLs. To prevent slow sequential HTTP requests, posters are fetched in parallel using `ThreadPoolExecutor`:
- **Tier 1:** OMDb REST API (`http://www.omdbapi.com/?t={title}&apikey=trilogy`) with 1.8s timeout.
- **Tier 2 (Fallback):** Wikipedia Search & Summary REST API (`https://en.wikipedia.org/api/rest_v1/page/summary/{key}`).
- **Tier 3 (Fallback):** High-definition Unsplash cinema still image.
- **Acceleration:** 5 parallel worker threads reduce loading latency from 8+ seconds to under 1.5 seconds.

---

## 🎨 7. Frontend Presentation (`app.py`)

The user interface uses modern web aesthetics via custom injected CSS:

1. **Cinema Dark Glassmorphism:**
   - Card backgrounds: `linear-gradient(145deg, #1e293b, #0f172a)`
   - Border radius: `16px` with subtle glowing border `rgba(255, 255, 255, 0.1)`
   - Hover micro-interactions: Smooth vertical elevation (`transform: translateY(-6px)`) with purple glow shadow (`#a855f7`).
2. **Dynamic Visual Badges:**
   - ⚡ **Similarity Match:** Emerald green badge (`#34d399`) showing exact calculated match percentage.
   - ⭐ **TMDB Rating:** Amber star badge (`#fbbf24`) displaying critic rating.
   - 🟣 **Genre Tags:** Indigo pills (`#a5b4fc`) indicating categories.
3. **Interactive Modals:**
   - Each movie card includes an expandable drawer revealing full synopsis, director credit, cast list, and a direct link to watch the official movie trailer on YouTube.
4. **Sidebar Controls:**
   - Real-time recommendation count slider ($K = 3$ to $12$).
   - Live genre filter dropdown.
   - Real-time dataset & ML model architecture summary.

---

## 🚀 8. How to Run the Project Locally

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Git

### Setup Steps
```bash
# 1. Clone the repository
git clone https://github.com/Nandinimane511/movie-recommender-system.git
cd movie-recommender-system

# 2. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate      # Windows
source .venv/bin/activate       # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Re-train the model pipeline
python preprocess.py

# 5. Launch the Web Application
streamlit run app.py
```

The web application will open automatically at **`http://localhost:8501`**.

---

## 📂 9. Project Directory Structure

```text
movie-recommender-system/
├── .venv/                                                      # Virtual environment (dependencies)
├── model/                                                      # Serialized trained model weights
│   ├── movie_list.pkl                                          # Preprocessed metadata DataFrame
│   ├── similarity.pkl                                          # Precomputed Cosine Similarity Matrix
│   └── knn_model.pkl                                           # Trained K-Nearest Neighbors ML Model
├── visualizations/                                             # High-resolution chart figures
│   ├── Fig_3_1_Correlation_Matrix.png                          # Correlation Heatmap
│   └── Fig_3_2_Cosine_Similarity_Matrix.png                    # Cosine Similarity Heatmap
├── tmdb_5000_movies.csv                                        # Raw TMDB Movies Dataset
├── tmdb_5000_credits.csv                                       # Raw TMDB Credits Dataset
├── Movie_Recommendation_System_EDA.ipynb                       # Complete, fully executed Jupyter Notebook
├── app.py                                                      # Production Streamlit Web App (Backend + Frontend)
├── preprocess.py                                               # Automated training pipeline script
├── requirements.txt                                            # Python dependencies
├── README.md                                                   # GitHub repository overview
├── DOCUMENTATION.md                                            # 📄 This complete system documentation file
└── CineMatch_AI_System_Architecture_and_Working_Guide.docx      # Comprehensive Word Report Document
```

---

*CineMatch AI — Developed by Nandini Mane*
