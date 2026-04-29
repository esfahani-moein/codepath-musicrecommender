# 🎧 VibeFinder 2.0 — AI-Powered Music Recommender with RAG & Reliability Testing

> **A production-ready simulation of a music recommendation system featuring Retrieval-Augmented Generation (RAG), multi-mode scoring, diversity controls, and a comprehensive reliability testing dashboard.**

---

## What's New in VibeFinder 2.0

This release is a complete redesign of the original recommender. Every core component has been upgraded with AI features, richer context, and measurable reliability.

### 1. Retrieval-Augmented Generation (RAG) Engine

Instead of only matching raw CSV features, the system now queries a **music knowledge base** before making recommendations. The knowledge base lives in `contents/` and includes:

- **`genres.json`** — Detailed descriptions, typical energy/valence ranges, related genres, key instruments, origins, and cultural notes for 15 genres.
- **`moods.json`** — Mood definitions, recommended genres, energy/valence ranges, activity contexts, and playlist examples for 14 moods.
- **`artists.json`** — Artist profiles with signature sounds, typical feature values, best-fit moods, and notable tracks.
- **`suggestions.json`** — Contextual suggestion cards for activities like "Gym & Workout," "Study & Work," "Evening Wind Down," and "Rainy Day Contemplation."

**How RAG works:** When a user asks for recommendations, the system:
1. Builds a TF-IDF index over all knowledge documents using `sklearn`.
2. Retrieves the most relevant genre, mood, artist, and suggestion documents for the user's query.
3. Uses retrieved knowledge to enable **fuzzy genre matching** (e.g., "indie pop" partially matches "pop") and **mood-genre alignment bonuses**.
4. Enriches every recommendation explanation with cultural context from the knowledge base.

### 2. Enhanced Recommendation Engine (`src/recommender_v2.py`)

- **Fuzzy Genre Matching**: Exact matches still get full points, but related genres (discovered via RAG) now earn partial credit.
- **RAG-Enhanced Scoring Mode**: A dedicated mode that balances all signals while rewarding genre-mood alignments validated by the knowledge base.
- **Diversity Penalty**: Prevents the top-k list from being dominated by a single artist or genre.
- **Novelty Boost**: Rewards genre/artist variety in recommendation lists.
- **Five Scoring Modes**: `balanced`, `genre_first`, `mood_first`, `energy_focused`, `rag_enhanced`.

### 3. Reliability & Testing System (`src/evaluation.py`)

A full quantitative testing suite to measure and compare recommender performance:

| Metric | Description |
|--------|-------------|
| **Intra-List Diversity** | Average pairwise Euclidean distance between recommended songs (higher = more varied) |
| **Relevance Score** | Normalized alignment of top-k results to user genre, mood, energy, and valence |
| **Catalog Coverage** | Percentage of the catalog that appears across all recommendation runs |
| **Genre Coverage** | Percentage of all genres represented across recommendations |
| **Novelty Score** | Jaccard distance comparing a mode's output to a baseline (higher = more different) |
| **Consistency** | Standard deviation of relevance scores across modes for the same profile |
| **A/B Testing** | Side-by-side comparison of two scoring modes with automatic winner selection |
| **Full System Audit** | End-to-end audit across all profiles and modes with a printable report |

### 4. Fancy Streamlit Dashboard (`app.py`)

A dark-themed, multi-tab interactive UI replacing the original CLI output:

- **🎯 Smart Recommender** — Build your vibe profile with sliders and dropdowns, toggle RAG enrichment, choose scoring modes, and see real-time recommendations with knowledge context.
- **🔍 RAG Explorer** — Search the knowledge base interactively and browse genres, moods, artists, and suggestions by category.
- **🧪 Testing & Reliability** — Run evaluations, view bar charts of diversity/relevance per profile, run A/B tests, and print a full system audit.
- **🎼 Catalog Browser** — Filter the song catalog by genre/mood/energy, explore a scatter plot of the song landscape, and view rich knowledge context per track.
- **🎭 Preset Profiles** — Try five built-in personas (High-Energy Pop, Chill Lofi, Deep Intense Rock, Conflicting Profile, Acoustic Jazz Lover) with side-by-side mode comparisons.

### 5. Comprehensive Test Suite

Tests are split across three modules:

- `tests/test_recommender.py` — Original tests (backward compatible)
- `tests/test_recommender_v2.py` — Tests for fuzzy matching, RAG-enhanced scoring, diversity penalties, and the new `Recommender` class
- `tests/test_rag_engine.py` — Tests for TF-IDF retrieval, knowledge base loading, related genres, and mood guidance
- `tests/test_evaluation.py` — Tests for diversity, coverage, relevance, novelty, A/B testing, and system audits

---

## Project Structure

```
project05_codepath_music_recommender/
├── app.py                          # Fancy Streamlit dashboard (main entry point)
├── src/
│   ├── main.py                     # Original CLI runner (still works)
│   ├── recommender.py              # Original v1 scoring logic (preserved)
│   ├── recommender_v2.py           # Enhanced v2 with RAG, fuzzy matching, 5 modes
│   ├── rag_engine.py               # TF-IDF retrieval over contents/ knowledge base
│   └── evaluation.py               # Diversity, relevance, coverage, A/B testing, audits
├── tests/
│   ├── test_recommender.py         # Original v1 tests
│   ├── test_recommender_v2.py      # v2 + RAG tests
│   ├── test_rag_engine.py          # Knowledge base retrieval tests
│   └── test_evaluation.py          # Reliability system tests
├── contents/
│   ├── genres.json                 # 15 genre knowledge documents
│   ├── moods.json                  # 14 mood knowledge documents
│   ├── artists.json                # 16 artist knowledge documents
│   └── suggestions.json            # 10 activity suggestion cards
├── data/
│   └── songs.csv                   # 18-track catalog
├── conftest.py                     # pytest path setup
├── requirements.txt                # Python dependencies
├── model_card.md                   # Model card for v1 (preserved)
├── reflection.md                   # Original reflection (preserved)
└── README.md                       # This file
```

---

## How The System Works

### VibeFinder 2.0 Architecture

```
User Prefs ──┬──► [RAG Engine] ──► Retrieve genre/mood/artist/suggestion docs
             │                           │
             │                           ▼
             │                    Enrich user profile
             │                    (related genres, mood guidance)
             │                           │
             └──► [Recommender v2] ◄─────┘
                          │
                          ▼
              For each song: score_song_rag()
                 - exact genre match
                 - related genre match (via RAG)
                 - mood match
                 - energy/valence/dance similarity
                 - acoustic bonus
                 - RAG mood-genre alignment bonus
                          │
                          ▼
              Apply diversity / novelty modifiers
                          │
                          ▼
              Sort & return Top K + explanations + knowledge context
```

### RAG Retrieval Pipeline

```
Knowledge Documents (contents/*.json)
         │
         ▼
    Flatten to text strings
         │
         ▼
    TF-IDF Vectorizer (sklearn)
         │
         ▼
    Sparse Document Matrix
         │
    ┌─────────────────┐
    │  User Query     │
    └────────┬────────┘
             ▼
    Cosine Similarity
             │
             ▼
    Top-K Retrieved Documents
             │
    ┌────────┴────────┐
    ▼                 ▼
Related Genres    Mood Guidance
```

---

## Features Used

**Song features** (from `data/songs.csv`):
- `genre` — categorical (pop, lofi, rock, ambient, jazz, synthwave, indie pop, R&B, hip-hop, classical, country, metal, reggae, folk, EDM)
- `mood` — categorical (happy, chill, intense, relaxed, moody, focused, soulful, confident, melancholy, nostalgic, aggressive, uplifting, reflective, euphoric)
- `energy` — numerical 0.0–1.0
- `valence` — numerical 0.0–1.0
- `danceability` — numerical 0.0–1.0
- `acousticness` — numerical 0.0–1.0

**Scoring Modes (`SCORING_MODES`)**:

| Mode | Genre | Mood | Energy | Related Genre | Best For |
|------|-------|------|--------|---------------|----------|
| `balanced` | 2.0 | 1.0 | 1.5 | 1.0 | General use |
| `genre_first` | 3.0 | 0.5 | 1.0 | 1.5 | Genre loyalists |
| `mood_first` | 0.5 | 3.0 | 1.0 | 0.3 | Mood explorers |
| `energy_focused` | 1.0 | 0.5 | 3.0 | 0.5 | Activity-based |
| `rag_enhanced` | 1.5 | 1.5 | 1.5 | 1.5 | AI-guided discovery |

---

## Getting Started

### Prerequisites

- Python 3.10+
- Conda environment named `quantenv` with dependencies pre-installed

### Setup

1. Activate the environment:

   ```bash
   conda activate quantenv
   ```

2. Install dependencies (if not already present):

   ```bash
   pip install -r requirements.txt
   ```

### Run the Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

### Run the Original CLI

```bash
python -m src.main
```

### Run Tests

Run the full test suite:

```bash
pytest
```

Run specific test modules:

```bash
pytest tests/test_rag_engine.py -v
pytest tests/test_evaluation.py -v
pytest tests/test_recommender_v2.py -v
```

---

## RAG Knowledge Base Demo

You can inspect the knowledge base directly:

```python
from src.rag_engine import MusicKnowledgeBase

kb = MusicKnowledgeBase("contents")

# What genres are related to pop?
print(kb.get_related_genres("pop"))
# ['electropop', 'indie pop', 'synthwave']

# What does the AI know about the "chill" mood?
print(kb.get_mood_recommendations("chill"))
# {'recommended_genres': ['lofi', 'ambient', 'jazz', 'reggae'],
#  'energy_range': '0.1-0.5', ... }

# Retrieve documents for a user query
results = kb.retrieve("happy energetic pop workout", top_k=3)
for r in results:
    print(f"[{r.doc_type}] {r.title} (score={r.score:.3f})")
```

---

## Evaluation & Reliability Demo

```python
from src.evaluation import full_system_audit, run_ab_test
from src.recommender_v2 import load_songs
from src.rag_engine import RAGEnrichedRecommender

songs = load_songs("data/songs.csv")
rag = RAGEnrichedRecommender()

profiles = {
    "High-Energy Pop": {"genre": "pop", "mood": "happy", "energy": 0.85, "valence": 0.8, "danceability": 0.8, "likes_acoustic": False},
    "Chill Lofi": {"genre": "lofi", "mood": "chill", "energy": 0.35, "valence": 0.6, "danceability": 0.55, "likes_acoustic": True},
}

# Full audit across all modes
audit = full_system_audit(profiles, songs, rag_enricher=rag)
print(audit["catalog_coverage"])   # e.g., 0.72
print(audit["best_mode_by_relevance"])  # e.g., "rag_enhanced"

# A/B test two modes
ab = run_ab_test(profiles, songs, mode_a="balanced", mode_b="rag_enhanced", rag_enricher_b=rag)
print(ab.winner_overall)  # "rag_enhanced"
```

---

## Key Design Decisions

1. **TF-IDF over dense embeddings**: We use `sklearn.feature_extraction.text.TfidfVectorizer` + cosine similarity instead of sentence transformers. This keeps the system lightweight, CPU-only, and fully compatible with the `quantenv` environment without requiring GPU or large model downloads.

2. **Modular RAG integration**: The RAG engine (`rag_engine.py`) is decoupled from the scorer (`recommender_v2.py`). You can swap in a dense embedding retriever (e.g., `sentence-transformers`) later without changing the scoring logic.

3. **Backward compatibility**: `src/recommender.py` and `tests/test_recommender.py` remain untouched so the original CLI and all v1 tests continue to pass.

4. **Declarative knowledge base**: All music knowledge is stored as structured JSON in `contents/`. This makes it easy to edit, version-control, and expand without changing code.

---

## Limitations and Future Work

- **Catalog size**: 18 songs is a simulation. A real system would use a vector database over millions of tracks.
- **No collaborative filtering**: We remain content-based. Adding user-item interaction data would enable hybrid recommendations.
- **Static knowledge base**: The `contents/` JSONs are hand-authored. Future versions could generate them from Wikipedia, Spotify APIs, or LLM summaries.
- **No listening history**: Temporal modeling and session-based recommendations are not yet implemented.
- **No audio features**: We use metadata only. Integrating spectrogram embeddings or audio feature extraction would improve relevance.

---

## License

MIT License — feel free to fork, extend, and publish your own music discovery tools.

---

## Credits

- **VibeFinder 1.0**: Original content-based scoring system with weighted feature matching.
- **VibeFinder 2.0**: RAG engine, fuzzy matching, evaluation framework, and Streamlit dashboard designed for publication and classroom exploration.

Built for the CodePath AI Engineering course. Ready for GitHub.
