"""
VibeFinder 2.0 - Interactive Music Recommender Dashboard
A fancy, multi-tab Streamlit app with RAG integration, visualizations,
and a built-in reliability testing dashboard.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.recommender_v2 import (
    load_songs,
    recommend_songs,
    score_song_rag,
    SCORING_MODES,
    format_recommendations_table,
    get_rich_explanation,
    Song,
    UserProfile,
    Recommender,
)
from src.rag_engine import MusicKnowledgeBase, RAGEnrichedRecommender, RetrievedDocument
from src.evaluation import (
    evaluate_profile,
    run_ab_test,
    full_system_audit,
    format_audit_report,
    intra_list_diversity,
    relevance_score,
)

st.set_page_config(
    page_title="VibeFinder 2.0",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def get_knowledge_base():
    return MusicKnowledgeBase("contents")

@st.cache_resource
def get_rag_enricher():
    kb = get_knowledge_base()
    return RAGEnrichedRecommender(kb)

@st.cache_data
def get_songs():
    return load_songs("data/songs.csv")

# Custom CSS for a premium feel
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #333;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid #4ECDC4;
        transition: transform 0.2s;
    }
    .recommendation-card:hover {
        transform: translateX(5px);
    }
    .score-badge {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        color: white;
        border-radius: 20px;
        padding: 0.3rem 0.8rem;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e1e2e;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #aaa;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2d2d44 !important;
        color: #fff !important;
        border-bottom: 2px solid #4ECDC4;
    }
    .rag-context {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 0.85rem;
        color: #ccc;
        border-left: 3px solid #FF6B6B;
    }
</style>
""", unsafe_allow_html=True)

def render_header():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<div class="main-header">🎧 VibeFinder 2.0</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">AI-Powered Music Discovery with RAG & Reliability Testing</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="text-align:right; padding-top:1rem;">
            <span style="background:#4ECDC4; color:#000; padding:0.3rem 0.7rem; border-radius:20px; font-size:0.8rem; font-weight:bold;">RAG ENABLED</span>
            <span style="background:#FF6B6B; color:#fff; padding:0.3rem 0.7rem; border-radius:20px; font-size:0.8rem; font-weight:bold; margin-left:0.5rem;">TESTING SYSTEM</span>
        </div>
        """, unsafe_allow_html=True)

# ---- Sidebar ----
def render_sidebar():
    with st.sidebar:
        st.markdown("### 🎛️ System Status")
        songs = get_songs()
        kb = get_knowledge_base()
        st.metric("Catalog Size", len(songs))
        st.metric("Knowledge Docs", len(kb.documents))
        genres = sorted(list({s["genre"] for s in songs}))
        st.markdown(f"**Genres:** {', '.join(genres)}")

        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        st.markdown("- **Scoring Modes:** 5")
        st.markdown("- **RAG Retrieval:** TF-IDF + Cosine")
        st.markdown("- **Test Metrics:** 6+")
        st.markdown("- **Eval Profiles:** 5 built-in")

        st.markdown("---")
        st.markdown("### 🧪 Run Audit")
        if st.button("🔬 Full System Audit", type="primary", width='stretch'):
            st.session_state.run_audit = True

# ---- Tab 1: Smart Recommender ----
def tab_recommender():
    st.markdown("## 🎯 Smart Recommender")
    st.markdown("Build your vibe profile and get AI-powered recommendations enriched with music knowledge.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Your Vibe Profile")
        with st.container(border=True):
            genre = st.selectbox("Genre", ["pop", "lofi", "rock", "ambient", "jazz", "synthwave", "indie pop", "R&B", "hip-hop", "classical", "country", "metal", "reggae", "folk", "EDM"])
            mood = st.selectbox("Mood", ["happy", "chill", "intense", "relaxed", "moody", "focused", "soulful", "confident", "melancholy", "nostalgic", "aggressive", "uplifting", "reflective", "euphoric"])
            energy = st.slider("Energy", 0.0, 1.0, 0.7, 0.05)
            valence = st.slider("Valence (Positivity)", 0.0, 1.0, 0.7, 0.05)
            danceability = st.slider("Danceability", 0.0, 1.0, 0.6, 0.05)
            likes_acoustic = st.toggle("Prefer Acoustic", value=False)

            mode = st.selectbox("Scoring Mode", list(SCORING_MODES.keys()), format_func=lambda x: x.replace("_", " ").title())
            use_rag = st.toggle("🔮 Use RAG Enrichment", value=True)
            diversity_penalty = st.toggle("🌈 Diversity Penalty", value=False)
            k = st.slider("Recommendations", 1, 10, 5)

            run_btn = st.button("✨ Get Recommendations", type="primary", width='stretch')

    user_prefs = {
        "genre": genre,
        "mood": mood,
        "energy": energy,
        "valence": valence,
        "danceability": danceability,
        "likes_acoustic": likes_acoustic,
    }

    with col2:
        if run_btn or st.session_state.get("auto_run", False):
            st.session_state.auto_run = False
            songs = get_songs()
            rag = get_rag_enricher() if use_rag else None

            # RAG insights
            if use_rag:
                with st.expander("🔍 RAG Insights", expanded=True):
                    related = rag.kb.get_related_genres(genre)
                    if related:
                        st.markdown(f"**Related genres to '{genre}':** {', '.join(related)}")
                    mood_info = rag.kb.get_mood_recommendations(mood)
                    if mood_info:
                        st.markdown(f"**Mood guidance:** {mood_info.get('explanation', '')[:200]}...")
                        st.markdown(f"**Suggested activities:** {', '.join(mood_info.get('activity_contexts', [])[:3])}")

            recs = recommend_songs(user_prefs, songs, k=k, mode=mode, diversity_penalty=diversity_penalty, rag_enricher=rag)

            # Metrics
            div = intra_list_diversity(recs)
            rel = relevance_score(user_prefs, recs)

            m1, m2, m3 = st.columns(3)
            m1.metric("Diversity", f"{div:.2f}", delta="Higher = more varied" if div > 0.3 else None)
            m2.metric("Relevance", f"{rel:.2f}", delta="Higher = better match" if rel > 0.5 else None)
            m3.metric("Results", len(recs))

            st.markdown("### 🎵 Your Recommendations")
            for i, (song, score, explanation) in enumerate(recs, 1):
                with st.container():
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <strong style="font-size:1.1rem; color:#fff;">{i}. {song['title']}</strong>
                                <span style="color:#aaa; margin-left:0.5rem;">by {song['artist']}</span>
                                <span style="margin-left:0.5rem; background:#333; color:#4ECDC4; padding:0.15rem 0.5rem; border-radius:4px; font-size:0.75rem;">{song['genre']}</span>
                                <span style="margin-left:0.3rem; background:#333; color:#FF6B6B; padding:0.15rem 0.5rem; border-radius:4px; font-size:0.75rem;">{song['mood']}</span>
                            </div>
                            <div class="score-badge">{score:.2f}</div>
                        </div>
                        <div style="margin-top:0.5rem; color:#bbb; font-size:0.85rem;">
                            {explanation}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if use_rag:
                        with st.expander("📚 Knowledge Context"):
                            ctx = get_rich_explanation(song, rag)
                            st.markdown(f'<div class="rag-context">{ctx}</div>', unsafe_allow_html=True)

            # Radar chart of top recommendation features vs user prefs
            if recs:
                top_song = recs[0][0]
                categories = ["Energy", "Valence", "Danceability", "Acousticness"]
                user_vals = [energy, valence, danceability, 0.8 if likes_acoustic else 0.2]
                song_vals = [top_song["energy"], top_song["valence"], top_song["danceability"], top_song["acousticness"]]

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=user_vals + [user_vals[0]],
                    theta=categories + [categories[0]],
                    fill='toself',
                    name='Your Profile',
                    line_color='#4ECDC4'
                ))
                fig.add_trace(go.Scatterpolar(
                    r=song_vals + [song_vals[0]],
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=f"'{top_song['title']}'",
                    line_color='#FF6B6B'
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    title="Profile vs Top Song",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#fff"),
                    height=350,
                )
                st.plotly_chart(fig, width='stretch')

# ---- Tab 2: RAG Knowledge Explorer ----
def tab_rag_explorer():
    st.markdown("## 🔍 RAG Knowledge Explorer")
    st.markdown("Search the music knowledge base and see what the AI retrieves to inform recommendations.")

    kb = get_knowledge_base()

    query = st.text_input("Search the knowledge base...", placeholder="e.g., 'happy energetic workout pop'")
    top_k = st.slider("Results to show", 1, 10, 3)

    if query:
        results = kb.retrieve(query, top_k=top_k)
        for r in results:
            with st.container(border=True):
                cols = st.columns([3, 1])
                with cols[0]:
                    st.markdown(f"**{r.title}**  <span style='color:#888; font-size:0.8rem;'>({r.doc_type})</span>", unsafe_allow_html=True)
                with cols[1]:
                    st.markdown(f"<div style='text-align:right; color:#4ECDC4; font-weight:bold;'>Score: {r.score:.3f}</div>", unsafe_allow_html=True)
                st.markdown(f"```\n{r.text[:500]}\n```")

    st.markdown("---")
    st.markdown("### 🗂️ Browse by Category")
    category = st.selectbox("Category", ["genres", "moods", "artists", "suggestions"])

    docs = [d for d in kb.documents if d["doc_type"] == category]
    for doc in docs:
        with st.expander(doc["title"]):
            st.markdown(doc["text"])

# ---- Tab 3: Testing & Reliability Dashboard ----
def tab_testing():
    st.markdown("## 🧪 Testing & Reliability Dashboard")
    st.markdown("Measure how well the recommender performs across profiles, modes, and metrics.")

    songs = get_songs()
    rag = get_rag_enricher()

    profiles = {
        "High-Energy Pop": {"genre": "pop", "mood": "happy", "energy": 0.85, "valence": 0.8, "danceability": 0.8, "likes_acoustic": False},
        "Chill Lofi": {"genre": "lofi", "mood": "chill", "energy": 0.35, "valence": 0.6, "danceability": 0.55, "likes_acoustic": True},
        "Deep Intense Rock": {"genre": "rock", "mood": "intense", "energy": 0.9, "valence": 0.45, "danceability": 0.6, "likes_acoustic": False},
        "Conflicting Profile": {"genre": "lofi", "mood": "intense", "energy": 0.9, "valence": 0.3, "danceability": 0.7, "likes_acoustic": False},
        "Acoustic Jazz Lover": {"genre": "jazz", "mood": "relaxed", "energy": 0.4, "valence": 0.7, "danceability": 0.5, "likes_acoustic": True},
    }

    mode = st.selectbox("Select mode to evaluate", list(SCORING_MODES.keys()), format_func=lambda x: x.replace("_", " ").title())
    use_rag_eval = st.toggle("Enable RAG in evaluation", value=True)
    run_eval = st.button("📊 Run Evaluation", type="primary")

    if run_eval or st.session_state.get("run_audit"):
        st.session_state.run_audit = False

        # Per-profile evaluation
        eval_results = []
        for name, prefs in profiles.items():
            ev = evaluate_profile(name, prefs, songs, mode=mode, k=5, rag_enricher=rag if use_rag_eval else None)
            eval_results.append(ev)

        # Summary metrics
        avg_div = sum(e.diversity_score for e in eval_results) / len(eval_results)
        avg_rel = sum(e.relevance_score for e in eval_results) / len(eval_results)
        avg_score = sum(e.avg_score for e in eval_results) / len(eval_results)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Diversity", f"{avg_div:.3f}")
        c2.metric("Avg Relevance", f"{avg_rel:.3f}")
        c3.metric("Avg Score", f"{avg_score:.2f}")
        c4.metric("Profiles Tested", len(profiles))

        # Per-profile bar chart
        df = pd.DataFrame([
            {
                "Profile": e.profile_name,
                "Diversity": e.diversity_score,
                "Relevance": e.relevance_score,
                "Avg Score": e.avg_score,
            }
            for e in eval_results
        ])

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Diversity by Profile", "Relevance by Profile"))
        fig.add_trace(go.Bar(x=df["Profile"], y=df["Diversity"], marker_color="#4ECDC4", name="Diversity"), row=1, col=1)
        fig.add_trace(go.Bar(x=df["Profile"], y=df["Relevance"], marker_color="#FF6B6B", name="Relevance"), row=1, col=2)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#fff"),
            showlegend=False,
            height=400,
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, width='stretch')

        # Detailed table
        st.markdown("### 📋 Detailed Results")
        for e in eval_results:
            with st.expander(f"{e.profile_name}  |  Diversity: {e.diversity_score:.3f}  |  Relevance: {e.relevance_score:.3f}"):
                st.markdown(f"**Top genres:** {', '.join(e.top_genres)}")
                st.markdown(f"**Top artists:** {', '.join(e.top_artists)}")
                st.markdown("**Explanations:**")
                for exp in e.explanations:
                    st.markdown(f"- {exp[:200]}")

        # A/B Test
        st.markdown("---")
        st.markdown("### 🆚 A/B Test: Balanced vs RAG-Enhanced")
        ab = run_ab_test(profiles, songs, mode_a="balanced", mode_b="rag_enhanced", k=5, rag_enricher_b=rag)

        ab_df = pd.DataFrame([
            {
                "Metric": "Diversity",
                "Balanced": ab.mode_a_avg_diversity,
                "RAG-Enhanced": ab.mode_b_avg_diversity,
            },
            {
                "Metric": "Relevance",
                "Balanced": ab.mode_a_avg_relevance,
                "RAG-Enhanced": ab.mode_b_avg_relevance,
            },
            {
                "Metric": "Avg Score",
                "Balanced": ab.mode_a_avg_score,
                "RAG-Enhanced": ab.mode_b_avg_score,
            },
        ])

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=ab_df["Metric"], y=ab_df["Balanced"], name="Balanced", marker_color="#45B7D1"))
        fig2.add_trace(go.Bar(x=ab_df["Metric"], y=ab_df["RAG-Enhanced"], name="RAG-Enhanced", marker_color="#FF6B6B"))
        fig2.update_layout(
            barmode="group",
            title="A/B Test Comparison",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#fff"),
            height=400,
        )
        st.plotly_chart(fig2, width='stretch')

        winners = {
            "Diversity": ab.winner_diversity,
            "Relevance": ab.winner_relevance,
            "Overall": ab.winner_overall,
        }
        cols = st.columns(3)
        for i, (metric, winner) in enumerate(winners.items()):
            color = "#4ECDC4" if winner == "rag_enhanced" else "#FF6B6B"
            cols[i].markdown(f"""
            <div style="text-align:center; background:{color}22; border:1px solid {color}; border-radius:12px; padding:1rem;">
                <div style="font-size:0.8rem; color:#888;">{metric} Winner</div>
                <div style="font-size:1.3rem; font-weight:bold; color:{color};">{winner.replace('_', ' ').title()}</div>
            </div>
            """, unsafe_allow_html=True)

        # Full system audit
        st.markdown("---")
        st.markdown("### 🔬 Full System Audit")
        audit = full_system_audit(profiles, songs, modes=list(SCORING_MODES.keys()), rag_enricher=rag)
        st.text(format_audit_report(audit))

# ---- Tab 4: Catalog Browser ----
def tab_catalog():
    st.markdown("## 🎼 Catalog Browser")
    st.markdown("Explore the full music catalog with interactive filters and visualizations.")

    songs = get_songs()
    rag = get_rag_enricher()
    df = pd.DataFrame(songs)

    col1, col2, col3 = st.columns(3)
    with col1:
        filter_genre = st.multiselect("Filter by Genre", sorted(df["genre"].unique()), default=list(df["genre"].unique()))
    with col2:
        filter_mood = st.multiselect("Filter by Mood", sorted(df["mood"].unique()), default=list(df["mood"].unique()))
    with col3:
        min_energy = st.slider("Min Energy", 0.0, 1.0, 0.0)

    filtered = df[(df["genre"].isin(filter_genre)) & (df["mood"].isin(filter_mood)) & (df["energy"] >= min_energy)]
    st.markdown(f"Showing **{len(filtered)}** of {len(df)} songs")

    # Scatter plot
    fig = px.scatter(
        filtered,
        x="energy",
        y="valence",
        color="genre",
        size="danceability",
        hover_data=["title", "artist", "mood", "tempo_bpm"],
        title="Song Landscape: Energy vs Valence",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#fff"),
    )
    st.plotly_chart(fig, width='stretch')

    # Song cards
    st.markdown("### 🎵 Songs")
    for _, song in filtered.iterrows():
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**{song['title']}** by *{song['artist']}*")
                st.markdown(f"`{song['genre']}` | `{song['mood']}` | Energy: {song['energy']:.2f} | Valence: {song['valence']:.2f}")
            with c2:
                if st.button("🔍 Explore", key=f"explore_{song['id']}"):
                    ctx = get_rich_explanation(song.to_dict(), rag)
                    st.session_state[f"ctx_{song['id']}"] = ctx
            if st.session_state.get(f"ctx_{song['id']}"):
                st.markdown(f'<div class="rag-context">{st.session_state[f"ctx_{song["id"]}"]}</div>', unsafe_allow_html=True)

# ---- Tab 5: Preset Profiles ----
def tab_presets():
    st.markdown("## 🎭 Preset Vibe Profiles")
    st.markdown("Try pre-built user personas and see how the system adapts.")

    songs = get_songs()
    rag = get_rag_enricher()

    presets = {
        "High-Energy Pop": {"genre": "pop", "mood": "happy", "energy": 0.85, "valence": 0.8, "danceability": 0.8, "likes_acoustic": False},
        "Chill Lofi": {"genre": "lofi", "mood": "chill", "energy": 0.35, "valence": 0.6, "danceability": 0.55, "likes_acoustic": True},
        "Deep Intense Rock": {"genre": "rock", "mood": "intense", "energy": 0.9, "valence": 0.45, "danceability": 0.6, "likes_acoustic": False},
        "Conflicting Profile": {"genre": "lofi", "mood": "intense", "energy": 0.9, "valence": 0.3, "danceability": 0.7, "likes_acoustic": False},
        "Acoustic Jazz Lover": {"genre": "jazz", "mood": "relaxed", "energy": 0.4, "valence": 0.7, "danceability": 0.5, "likes_acoustic": True},
    }

    selected = st.selectbox("Choose a persona", list(presets.keys()))
    prefs = presets[selected]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Profile Details")
        st.json(prefs)
        related = rag.kb.get_related_genres(prefs["genre"])
        if related:
            st.markdown(f"**Related genres (via RAG):** {', '.join(related)}")
        mood_info = rag.kb.get_mood_recommendations(prefs["mood"])
        if mood_info:
            st.markdown(f"**Mood guidance:** {mood_info.get('explanation', '')[:150]}...")

    with col2:
        st.markdown("### Recommendations (Balanced + RAG)")
        recs = recommend_songs(prefs, songs, k=5, mode="rag_enhanced", rag_enricher=rag)
        for i, (song, score, explanation) in enumerate(recs, 1):
            st.markdown(f"""
            <div class="recommendation-card">
                <strong>{i}. {song['title']}</strong> <span style="color:#888;">by {song['artist']}</span>
                <span style="margin-left:0.5rem; background:#333; color:#4ECDC4; padding:0.1rem 0.4rem; border-radius:4px; font-size:0.7rem;">{song['genre']}</span>
                <div style="margin-top:0.3rem; font-size:0.8rem; color:#aaa;">{explanation}</div>
            </div>
            """, unsafe_allow_html=True)

    # Compare modes side-by-side
    st.markdown("---")
    st.markdown("### ⚖️ Mode Comparison")
    cols = st.columns(len(SCORING_MODES))
    for i, mode in enumerate(SCORING_MODES.keys()):
        with cols[i]:
            st.markdown(f"**{mode.replace('_', ' ').title()}**")
            mode_recs = recommend_songs(prefs, songs, k=3, mode=mode, rag_enricher=rag)
            for j, (song, score, _) in enumerate(mode_recs, 1):
                st.markdown(f"{j}. **{song['title']}** ({score:.2f})")

# ---- Main ----
def main():
    render_header()
    render_sidebar()

    tabs = st.tabs([
        "🎯 Smart Recommender",
        "🔍 RAG Explorer",
        "🧪 Testing & Reliability",
        "🎼 Catalog Browser",
        "🎭 Preset Profiles",
    ])

    with tabs[0]:
        tab_recommender()
    with tabs[1]:
        tab_rag_explorer()
    with tabs[2]:
        tab_testing()
    with tabs[3]:
        tab_catalog()
    with tabs[4]:
        tab_presets()

if __name__ == "__main__":
    main()
