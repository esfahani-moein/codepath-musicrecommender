"""
Tests for VibeFinder 2.0 enhanced recommender with RAG integration.
"""

import pytest
from src.recommender_v2 import (
    Song, UserProfile, Recommender,
    load_songs, score_song_rag, recommend_songs, SCORING_MODES, get_rich_explanation,
)
from src.rag_engine import RAGEnrichedRecommender


def make_small_recommender():
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)
    assert len(results) == 2
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]
    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


def test_score_song_rag_genre_match():
    user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}
    song = {"genre": "pop", "mood": "sad", "energy": 0.5, "valence": 0.5, "danceability": 0.5, "acousticness": 0.2}
    score, reasons = score_song_rag(user_prefs, song, mode="balanced", rag_enricher=None)
    assert score >= 2.0
    assert any("genre match" in r for r in reasons)


def test_score_song_rag_mood_match():
    user_prefs = {"genre": "rock", "mood": "chill", "energy": 0.4}
    song = {"genre": "lofi", "mood": "chill", "energy": 0.4, "valence": 0.5, "danceability": 0.5, "acousticness": 0.3}
    score, reasons = score_song_rag(user_prefs, song)
    assert any("mood match" in r for r in reasons)


def test_score_song_rag_energy_similarity():
    user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.5}
    song_close = {"genre": "rock", "mood": "sad", "energy": 0.5, "valence": 0.5, "danceability": 0.5, "acousticness": 0.2}
    song_far = {"genre": "rock", "mood": "sad", "energy": 0.9, "valence": 0.5, "danceability": 0.5, "acousticness": 0.2}
    score_close, _ = score_song_rag(user_prefs, song_close)
    score_far, _ = score_song_rag(user_prefs, song_far)
    assert score_close > score_far


def test_score_song_rag_acoustic_bonus():
    user_prefs = {"genre": "jazz", "mood": "relaxed", "energy": 0.4, "likes_acoustic": True}
    song = {"genre": "jazz", "mood": "relaxed", "energy": 0.4, "valence": 0.5, "danceability": 0.5, "acousticness": 0.9}
    score, reasons = score_song_rag(user_prefs, song)
    assert any("acoustic bonus" in r for r in reasons)


def test_score_song_rag_no_acoustic_bonus_when_not_preferred():
    user_prefs = {"genre": "jazz", "mood": "relaxed", "energy": 0.4, "likes_acoustic": False}
    song = {"genre": "jazz", "mood": "relaxed", "energy": 0.4, "valence": 0.5, "danceability": 0.5, "acousticness": 0.9}
    score, reasons = score_song_rag(user_prefs, song)
    assert not any("acoustic bonus" in r for r in reasons)


def test_load_songs_returns_list():
    songs = load_songs("data/songs.csv")
    assert isinstance(songs, list)
    assert len(songs) == 18


def test_load_songs_numerical_types():
    songs = load_songs("data/songs.csv")
    first = songs[0]
    assert isinstance(first["energy"], float)
    assert isinstance(first["tempo_bpm"], float)
    assert isinstance(first["id"], int)


def test_recommend_songs_returns_top_k():
    songs = load_songs("data/songs.csv")
    user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}
    results = recommend_songs(user_prefs, songs, k=3)
    assert len(results) == 3
    assert results[0][1] >= results[1][1] >= results[2][1]


def test_recommend_songs_format():
    songs = load_songs("data/songs.csv")
    user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}
    results = recommend_songs(user_prefs, songs, k=1)
    song, score, explanation = results[0]
    assert isinstance(score, float)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


def test_related_genre_match_with_rag():
    rag = RAGEnrichedRecommender()
    user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.7}
    song = {"genre": "indie pop", "mood": "happy", "energy": 0.7, "valence": 0.7, "danceability": 0.7, "acousticness": 0.3}
    score, reasons = score_song_rag(user_prefs, song, mode="balanced", rag_enricher=rag)
    matched = any("related genre" in r.lower() for r in reasons) or any("genre match" in r.lower() for r in reasons)
    assert matched


def test_rag_enhanced_mode_exists():
    assert "rag_enhanced" in SCORING_MODES


def test_diversity_penalty_reduces_duplicates():
    songs_data = [
        {"id": 1, "title": "A", "artist": "Artist1", "genre": "pop", "mood": "happy", "energy": 0.9, "valence": 0.9, "danceability": 0.9, "acousticness": 0.1},
        {"id": 2, "title": "B", "artist": "Artist1", "genre": "pop", "mood": "happy", "energy": 0.85, "valence": 0.85, "danceability": 0.85, "acousticness": 0.15},
        {"id": 3, "title": "C", "artist": "Artist2", "genre": "rock", "mood": "intense", "energy": 0.9, "valence": 0.5, "danceability": 0.7, "acousticness": 0.1},
    ]
    user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8, "likes_acoustic": False}
    recs_no_div = recommend_songs(user_prefs, songs_data, k=2, diversity_penalty=False)
    recs_div = recommend_songs(user_prefs, songs_data, k=2, diversity_penalty=True)
    genres_no_div = [r[0]["genre"] for r in recs_no_div]
    genres_div = [r[0]["genre"] for r in recs_div]
    # Diversity penalty should encourage variety
    assert len(set(genres_div)) >= len(set(genres_no_div))


def test_recommender_with_rag():
    songs = [
        Song(id=1, title="A", artist="Artist1", genre="pop", mood="happy", energy=0.9, tempo_bpm=120, valence=0.9, danceability=0.9, acousticness=0.1),
        Song(id=2, title="B", artist="Artist2", genre="indie pop", mood="happy", energy=0.8, tempo_bpm=110, valence=0.8, danceability=0.8, acousticness=0.2),
    ]
    rec = Recommender(songs)
    rag = RAGEnrichedRecommender()
    user = UserProfile(favorite_genre="pop", favorite_mood="happy", target_energy=0.85, likes_acoustic=False)
    results = rec.recommend(user, k=2, rag_enricher=rag)
    assert len(results) == 2
