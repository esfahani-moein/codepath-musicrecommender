"""
Tests for the evaluation and reliability system.
"""

import pytest
import math
from src.evaluation import (
    intra_list_diversity,
    catalog_coverage,
    genre_coverage,
    relevance_score,
    novelty_score,
    evaluate_profile,
    run_ab_test,
    full_system_audit,
    format_audit_report,
)
from src.recommender_v2 import load_songs, RAGEnrichedRecommender


@pytest.fixture
def songs():
    return load_songs("data/songs.csv")


@pytest.fixture
def rag():
    return RAGEnrichedRecommender()


@pytest.fixture
def simple_recs():
    return [
        ({"id": 1, "energy": 0.9, "valence": 0.9, "danceability": 0.9, "acousticness": 0.1, "genre": "pop", "mood": "happy"}, 5.0, "reason"),
        ({"id": 2, "energy": 0.1, "valence": 0.1, "danceability": 0.1, "acousticness": 0.9, "genre": "lofi", "mood": "chill"}, 4.0, "reason"),
    ]


@pytest.fixture
def similar_recs():
    return [
        ({"id": 1, "energy": 0.8, "valence": 0.8, "danceability": 0.8, "acousticness": 0.2, "genre": "pop", "mood": "happy"}, 5.0, "reason"),
        ({"id": 2, "energy": 0.82, "valence": 0.82, "danceability": 0.79, "acousticness": 0.18, "genre": "pop", "mood": "happy"}, 4.8, "reason"),
    ]


def test_intra_list_diversity_high_for_different_songs(simple_recs):
    div = intra_list_diversity(simple_recs)
    assert div > 0.3
    assert div <= 1.0


def test_intra_list_diversity_low_for_similar_songs(similar_recs):
    div = intra_list_diversity(similar_recs)
    assert div < 0.2


def test_intra_list_diversity_single_item():
    div = intra_list_diversity([({"id": 1, "energy": 0.5, "valence": 0.5, "danceability": 0.5, "acousticness": 0.5}, 1.0, "")])
    assert div == 0.0


def test_catalog_coverage(simple_recs, songs):
    coverage = catalog_coverage([simple_recs], total_catalog_size=len(songs))
    assert 0.0 <= coverage <= 1.0
    assert coverage == 2 / len(songs)


def test_genre_coverage(simple_recs):
    all_genres = ["pop", "lofi", "rock", "jazz"]
    cov = genre_coverage([simple_recs], all_genres)
    assert cov == 0.5


def test_relevance_score_perfect_match():
    recs = [
        ({"id": 1, "genre": "pop", "mood": "happy", "energy": 0.8, "valence": 0.8}, 5.0, ""),
    ]
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8, "valence": 0.8}
    rel = relevance_score(prefs, recs)
    assert rel > 0.9


def test_relevance_score_poor_match():
    recs = [
        ({"id": 1, "genre": "metal", "mood": "aggressive", "energy": 0.95, "valence": 0.2}, 5.0, ""),
    ]
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.3, "valence": 0.9}
    rel = relevance_score(prefs, recs)
    assert rel < 0.3


def test_novelty_score_identical():
    recs_a = [({"id": 1}, 1.0, ""), ({"id": 2}, 1.0, "")]
    recs_b = [({"id": 1}, 1.0, ""), ({"id": 2}, 1.0, "")]
    nov = novelty_score(recs_a, recs_b)
    assert nov == 0.0


def test_novelty_score_completely_different():
    recs_a = [({"id": 1}, 1.0, ""), ({"id": 2}, 1.0, "")]
    recs_b = [({"id": 3}, 1.0, ""), ({"id": 4}, 1.0, "")]
    nov = novelty_score(recs_a, recs_b)
    assert nov == 1.0


def test_evaluate_profile_returns_result(songs, rag):
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8, "valence": 0.8, "danceability": 0.8, "likes_acoustic": False}
    ev = evaluate_profile("Test", prefs, songs, mode="balanced", k=3, rag_enricher=rag)
    assert ev.profile_name == "Test"
    assert ev.mode == "balanced"
    assert 0.0 <= ev.diversity_score <= 1.0
    assert 0.0 <= ev.relevance_score <= 1.0
    assert len(ev.top_genres) == 3
    assert len(ev.explanations) == 3


def test_ab_test_returns_winner(songs, rag):
    profiles = {
        "Pop": {"genre": "pop", "mood": "happy", "energy": 0.8, "valence": 0.8, "likes_acoustic": False},
        "Chill": {"genre": "lofi", "mood": "chill", "energy": 0.3, "valence": 0.6, "likes_acoustic": True},
    }
    ab = run_ab_test(profiles, songs, mode_a="balanced", mode_b="rag_enhanced", k=3, rag_enricher_b=rag)
    assert ab.mode_a == "balanced"
    assert ab.mode_b == "rag_enhanced"
    assert ab.winner_diversity in ("balanced", "rag_enhanced")
    assert ab.winner_relevance in ("balanced", "rag_enhanced")
    assert ab.winner_overall in ("balanced", "rag_enhanced")
    assert len(ab.per_profile_results) == 2


def test_full_system_audit_returns_dict(songs, rag):
    profiles = {
        "Pop": {"genre": "pop", "mood": "happy", "energy": 0.8, "likes_acoustic": False},
        "Chill": {"genre": "lofi", "mood": "chill", "energy": 0.3, "likes_acoustic": True},
    }
    audit = full_system_audit(profiles, songs, modes=["balanced", "rag_enhanced"], rag_enricher=rag)
    assert "catalog_coverage" in audit
    assert "genre_coverage" in audit
    assert "avg_diversity" in audit
    assert "avg_relevance" in audit
    assert "best_mode_by_relevance" in audit
    assert "best_mode_by_diversity" in audit
    assert 0.0 <= audit["catalog_coverage"] <= 1.0


def test_format_audit_report_is_string(songs, rag):
    profiles = {
        "Pop": {"genre": "pop", "mood": "happy", "energy": 0.8, "likes_acoustic": False},
    }
    audit = full_system_audit(profiles, songs, modes=["balanced"], rag_enricher=rag)
    report = format_audit_report(audit)
    assert isinstance(report, str)
    assert "VibeFinder 2.0" in report
    assert "Catalog coverage" in report
