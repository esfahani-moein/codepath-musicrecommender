"""
Tests for the RAG (Retrieval-Augmented Generation) engine.
"""

import pytest
from src.rag_engine import MusicKnowledgeBase, RAGEnrichedRecommender, RetrievedDocument


@pytest.fixture
def kb():
    return MusicKnowledgeBase("contents")


@pytest.fixture
def rag(kb):
    return RAGEnrichedRecommender(kb)


def test_knowledge_base_loads_documents(kb):
    assert len(kb.documents) > 0
    genres = [d for d in kb.documents if d["doc_type"] == "genres"]
    assert len(genres) >= 10


def test_retrieve_returns_top_k(kb):
    results = kb.retrieve("happy pop energetic workout", top_k=3)
    assert len(results) == 3
    assert all(isinstance(r, RetrievedDocument) for r in results)
    assert all(r.score > 0 for r in results)


def test_retrieve_ordered_by_relevance(kb):
    results = kb.retrieve("jazz relaxing acoustic", top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_by_genre_finds_genre_docs(kb):
    results = kb.retrieve_by_genre("pop", top_k=2)
    assert len(results) > 0
    titles = [r.title.lower() for r in results]
    assert any("pop" in t for t in titles)


def test_retrieve_by_mood_finds_mood_docs(kb):
    results = kb.retrieve_by_mood("chill", top_k=2)
    assert len(results) > 0
    titles = [r.title.lower() for r in results]
    assert any("chill" in t for t in titles)


def test_get_related_genres_returns_list(kb):
    related = kb.get_related_genres("pop")
    assert isinstance(related, list)
    # Pop should have related genres like indie pop, synthwave, etc.
    assert len(related) > 0
    assert "pop" not in related


def test_get_mood_recommendations_returns_dict(kb):
    info = kb.get_mood_recommendations("happy")
    assert isinstance(info, dict)
    assert "recommended_genres" in info
    assert "energy_range" in info


def test_get_song_context_returns_string(kb):
    song = {"title": "Sunrise City", "artist": "Neon Echo", "genre": "pop", "mood": "happy"}
    ctx = kb.get_song_context(song)
    assert isinstance(ctx, str)
    assert len(ctx) > 0


def test_rag_enriched_recommender_enrich_user_prefs(rag):
    user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}
    enriched = rag.enrich_user_prefs(user_prefs)
    assert "related_genres" in enriched
    assert "mood_guidance" in enriched
    assert enriched["genre"] == "pop"


def test_rag_enriched_recommender_suggests_energy_for_mood(rag):
    user_prefs = {"genre": "pop", "mood": "happy"}
    enriched = rag.enrich_user_prefs(user_prefs)
    assert "suggested_energy" in enriched or "suggested_valence" in enriched


def test_generate_explanation_with_rag(rag):
    song = {"title": "Test", "artist": "Artist", "genre": "pop", "mood": "happy"}
    explanation = rag.generate_explanation_with_rag(
        {"genre": "pop", "mood": "happy"}, song, 3.5, ["genre match"]
    )
    assert isinstance(explanation, str)
    assert "Score:" in explanation
    assert "genre match" in explanation


def test_empty_query_returns_empty(kb):
    results = kb.retrieve("", top_k=3)
    assert len(results) == 3


def test_all_document_types_present(kb):
    types = set(d["doc_type"] for d in kb.documents)
    assert "genres" in types
    assert "moods" in types
    assert "artists" in types
    assert "suggestions" in types
