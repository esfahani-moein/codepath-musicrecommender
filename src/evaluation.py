"""
Reliability and Testing System for VibeFinder 2.0.

Provides metrics to measure recommendation quality:
- Diversity: intra-list diversity (variety within recommendations)
- Coverage: catalog coverage over multiple queries
- Relevance Score: how well recommendations match stated preferences
- A/B Testing framework: compare two scoring modes side-by-side
- Novelty: how unexpected vs baseline the recommendations are
- Consistency: whether the same profile gets similar results across runs
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
from typing import List, Dict, Tuple, Any, Callable
from dataclasses import dataclass, field
import numpy as np
from collections import Counter

from src.recommender_v2 import score_song_rag, recommend_songs, SCORING_MODES, RAGEnrichedRecommender


@dataclass
class EvaluationResult:
    """Result of a single evaluation run."""
    profile_name: str
    mode: str
    diversity_score: float
    relevance_score: float
    coverage_ratio: float
    avg_score: float
    explanations: List[str] = field(default_factory=list)
    top_genres: List[str] = field(default_factory=list)
    top_artists: List[str] = field(default_factory=list)


@dataclass
class ABTestResult:
    """Result of an A/B test comparing two modes."""
    mode_a: str
    mode_b: str
    mode_a_avg_diversity: float
    mode_b_avg_diversity: float
    mode_a_avg_relevance: float
    mode_b_avg_relevance: float
    mode_a_avg_score: float
    mode_b_avg_score: float
    winner_diversity: str
    winner_relevance: str
    winner_overall: str
    per_profile_results: List[Dict[str, Any]] = field(default_factory=list)


def _euclidean_distance(a: Dict, b: Dict, keys: List[str]) -> float:
    """Compute Euclidean distance between two song dicts over given keys."""
    total = 0.0
    for k in keys:
        total += (a.get(k, 0.0) - b.get(k, 0.0)) ** 2
    return math.sqrt(total)


def intra_list_diversity(recommendations: List[Tuple[Dict, float, str]], feature_keys: List[str] = None) -> float:
    """
    Measure diversity within a recommendation list using average pairwise distance.
    Higher = more diverse. Range roughly 0 to 1 for normalized features.
    """
    if feature_keys is None:
        feature_keys = ["energy", "valence", "danceability", "acousticness"]
    n = len(recommendations)
    if n < 2:
        return 0.0

    total_distance = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = _euclidean_distance(recommendations[i][0], recommendations[j][0], feature_keys)
            total_distance += d
            count += 1

    avg = total_distance / count if count else 0.0
    # Normalize by max possible distance in 4D unit hypercube = sqrt(4) = 2
    return min(avg / 2.0, 1.0)


def catalog_coverage(all_recommendations: List[List[Tuple[Dict, float, str]]], total_catalog_size: int) -> float:
    """Percentage of the catalog that appears in any recommendation list."""
    seen_ids = set()
    for rec_list in all_recommendations:
        for song, _, _ in rec_list:
            seen_ids.add(song.get("id", song.get("title", "")))
    return len(seen_ids) / total_catalog_size if total_catalog_size else 0.0


def genre_coverage(all_recommendations: List[List[Tuple[Dict, float, str]]], all_genres: List[str]) -> float:
    """Percentage of all genres represented across recommendations."""
    seen_genres = set()
    for rec_list in all_recommendations:
        for song, _, _ in rec_list:
            seen_genres.add(song.get("genre", "").lower())
    return len(seen_genres) / len(all_genres) if all_genres else 0.0


def relevance_score(user_prefs: Dict, recommendations: List[Tuple[Dict, float, str]], k: int = None) -> float:
    """
    Compute a normalized relevance score based on how well the top-k items match user prefs.
    Considers genre, mood, and energy alignment.
    """
    if k is None:
        k = len(recommendations)
    items = recommendations[:k]
    if not items:
        return 0.0

    scores = []
    user_genre = user_prefs.get("genre", "").lower()
    user_mood = user_prefs.get("mood", "").lower()
    user_energy = user_prefs.get("energy", 0.5)
    user_valence = user_prefs.get("valence", 0.5)

    for song, _, _ in items:
        match = 0.0
        if user_genre == song.get("genre", "").lower():
            match += 0.4
        if user_mood == song.get("mood", "").lower():
            match += 0.3
        energy_diff = abs(song.get("energy", 0.5) - user_energy)
        match += max(0.0, 0.2 * (1.0 - energy_diff))
        valence_diff = abs(song.get("valence", 0.5) - user_valence)
        match += max(0.0, 0.1 * (1.0 - valence_diff))
        scores.append(match)

    return float(np.mean(scores))


def novelty_score(recommendations: List[Tuple[Dict, float, str]], baseline_recommendations: List[Tuple[Dict, float, str]]) -> float:
    """
    Measure how 'novel' a recommendation list is compared to a baseline.
    Computes Jaccard distance between the two sets of song IDs.
    Higher = more different from baseline.
    """
    rec_ids = {s[0].get("id", s[0].get("title", "")) for s in recommendations}
    base_ids = {s[0].get("id", s[0].get("title", "")) for s in baseline_recommendations}
    intersection = len(rec_ids & base_ids)
    union = len(rec_ids | base_ids)
    if union == 0:
        return 0.0
    jaccard_sim = intersection / union
    return 1.0 - jaccard_sim


def evaluate_profile(
    profile_name: str,
    user_prefs: Dict,
    songs: List[Dict],
    mode: str = "balanced",
    k: int = 5,
    rag_enricher: Any = None,
    diversity_penalty: bool = False,
) -> EvaluationResult:
    """Run a full evaluation for a single user profile."""
    recs = recommend_songs(
        user_prefs, songs, k=k, mode=mode,
        diversity_penalty=diversity_penalty,
        rag_enricher=rag_enricher,
    )

    div = intra_list_diversity(recs)
    rel = relevance_score(user_prefs, recs)
    avg = float(np.mean([score for _, score, _ in recs])) if recs else 0.0
    genres = [s["genre"] for s, _, _ in recs]
    artists = [s["artist"] for s, _, _ in recs]
    cov_ratio = len(set(genres)) / k if k else 0.0

    return EvaluationResult(
        profile_name=profile_name,
        mode=mode,
        diversity_score=div,
        relevance_score=rel,
        coverage_ratio=cov_ratio,
        avg_score=avg,
        explanations=[exp for _, _, exp in recs],
        top_genres=genres,
        top_artists=artists,
    )


def run_ab_test(
    profiles: Dict[str, Dict],
    songs: List[Dict],
    mode_a: str = "balanced",
    mode_b: str = "rag_enhanced",
    k: int = 5,
    rag_enricher_a: Any = None,
    rag_enricher_b: Any = None,
) -> ABTestResult:
    """Compare two scoring modes across multiple user profiles."""
    results_a = []
    results_b = []
    per_profile = []

    for name, prefs in profiles.items():
        ev_a = evaluate_profile(name, prefs, songs, mode=mode_a, k=k, rag_enricher=rag_enricher_a)
        ev_b = evaluate_profile(name, prefs, songs, mode=mode_b, k=k, rag_enricher=rag_enricher_b)
        results_a.append(ev_a)
        results_b.append(ev_b)

        per_profile.append({
            "profile": name,
            f"{mode_a}_diversity": ev_a.diversity_score,
            f"{mode_b}_diversity": ev_b.diversity_score,
            f"{mode_a}_relevance": ev_a.relevance_score,
            f"{mode_b}_relevance": ev_b.relevance_score,
            f"{mode_a}_avg_score": ev_a.avg_score,
            f"{mode_b}_avg_score": ev_b.avg_score,
            "winner_diversity": mode_a if ev_a.diversity_score > ev_b.diversity_score else mode_b,
            "winner_relevance": mode_a if ev_a.relevance_score > ev_b.relevance_score else mode_b,
        })

    avg_div_a = float(np.mean([r.diversity_score for r in results_a]))
    avg_div_b = float(np.mean([r.diversity_score for r in results_b]))
    avg_rel_a = float(np.mean([r.relevance_score for r in results_a]))
    avg_rel_b = float(np.mean([r.relevance_score for r in results_b]))
    avg_score_a = float(np.mean([r.avg_score for r in results_a]))
    avg_score_b = float(np.mean([r.avg_score for r in results_b]))

    winner_div = mode_a if avg_div_a > avg_div_b else mode_b
    winner_rel = mode_a if avg_rel_a > avg_rel_b else mode_b
    # Overall winner based on average of normalized diversity + relevance
    overall_a = (avg_div_a + avg_rel_a) / 2.0
    overall_b = (avg_div_b + avg_rel_b) / 2.0
    winner_overall = mode_a if overall_a > overall_b else mode_b

    return ABTestResult(
        mode_a=mode_a,
        mode_b=mode_b,
        mode_a_avg_diversity=avg_div_a,
        mode_b_avg_diversity=avg_div_b,
        mode_a_avg_relevance=avg_rel_a,
        mode_b_avg_relevance=avg_rel_b,
        mode_a_avg_score=avg_score_a,
        mode_b_avg_score=avg_score_b,
        winner_diversity=winner_div,
        winner_relevance=winner_rel,
        winner_overall=winner_overall,
        per_profile_results=per_profile,
    )


def full_system_audit(
    profiles: Dict[str, Dict],
    songs: List[Dict],
    modes: List[str] = None,
    k: int = 5,
    rag_enricher: Any = None,
) -> Dict[str, Any]:
    """Run a comprehensive audit of the recommender system across all modes and profiles."""
    if modes is None:
        modes = list(SCORING_MODES.keys())

    all_evals = []
    all_recs = []

    for mode in modes:
        for name, prefs in profiles.items():
            ev = evaluate_profile(name, prefs, songs, mode=mode, k=k, rag_enricher=rag_enricher)
            all_evals.append(ev)
            recs = recommend_songs(prefs, songs, k=k, mode=mode, rag_enricher=rag_enricher)
            all_recs.append(recs)

    all_genres = list({s["genre"] for s in songs})
    catalog_cov = catalog_coverage(all_recs, len(songs))
    genre_cov = genre_coverage(all_recs, all_genres)

    # Consistency check: same profile should get similar results across modes
    consistency_scores = {}
    for name in profiles:
        mode_scores = []
        for mode in modes:
            ev = next((e for e in all_evals if e.profile_name == name and e.mode == mode), None)
            if ev:
                mode_scores.append(ev.relevance_score)
        if len(mode_scores) > 1:
            consistency_scores[name] = 1.0 - float(np.std(mode_scores))

    return {
        "num_profiles": len(profiles),
        "num_modes": len(modes),
        "catalog_coverage": catalog_cov,
        "genre_coverage": genre_cov,
        "avg_diversity": float(np.mean([e.diversity_score for e in all_evals])),
        "avg_relevance": float(np.mean([e.relevance_score for e in all_evals])),
        "avg_score": float(np.mean([e.avg_score for e in all_evals])),
        "consistency_by_profile": consistency_scores,
        "best_mode_by_relevance": _best_mode(all_evals, "relevance_score"),
        "best_mode_by_diversity": _best_mode(all_evals, "diversity_score"),
    }


def _best_mode(evals: List[EvaluationResult], metric: str) -> str:
    """Find the mode with highest average metric."""
    by_mode = {}
    counts = {}
    for e in evals:
        by_mode[e.mode] = by_mode.get(e.mode, 0.0) + getattr(e, metric, 0.0)
        counts[e.mode] = counts.get(e.mode, 0) + 1
    best = None
    best_val = -1.0
    for mode, total in by_mode.items():
        avg = total / counts[mode]
        if avg > best_val:
            best_val = avg
            best = mode
    return best


def format_audit_report(audit: Dict[str, Any]) -> str:
    """Pretty-print an audit report."""
    lines = [
        "=" * 70,
        "  VibeFinder 2.0 - System Reliability Audit",
        "=" * 70,
        f"Profiles tested:        {audit['num_profiles']}",
        f"Modes tested:           {audit['num_modes']}",
        f"Catalog coverage:       {audit['catalog_coverage']:.2%}",
        f"Genre coverage:         {audit['genre_coverage']:.2%}",
        f"Avg diversity:          {audit['avg_diversity']:.3f}",
        f"Avg relevance:          {audit['avg_relevance']:.3f}",
        f"Avg recommendation score: {audit['avg_score']:.3f}",
        f"Best mode (relevance):  {audit['best_mode_by_relevance']}",
        f"Best mode (diversity):  {audit['best_mode_by_diversity']}",
        "-" * 70,
        "Consistency by profile (1.0 = perfectly consistent across modes):",
    ]
    for profile, score in audit["consistency_by_profile"].items():
        lines.append(f"  {profile:<30} {score:.3f}")
    lines.append("=" * 70)
    return "\n".join(lines)


if __name__ == "__main__":
    from src.recommender_v2 import load_songs, RAGEnrichedRecommender
    songs = load_songs("data/songs.csv")
    profiles = {
        "Pop/Happy": {"genre": "pop", "mood": "happy", "energy": 0.8, "valence": 0.8, "likes_acoustic": False},
        "Chill/Lofi": {"genre": "lofi", "mood": "chill", "energy": 0.35, "valence": 0.6, "likes_acoustic": True},
        "Rock/Intense": {"genre": "rock", "mood": "intense", "energy": 0.9, "valence": 0.45, "likes_acoustic": False},
    }
    rag = RAGEnrichedRecommender()
    audit = full_system_audit(profiles, songs, rag_enricher=rag)
    print(format_audit_report(audit))

    ab = run_ab_test(profiles, songs, mode_a="balanced", mode_b="rag_enhanced", rag_enricher_b=rag)
    print("\nA/B Test: balanced vs rag_enhanced")
    print(f"  Diversity: {ab.mode_a}={ab.mode_a_avg_diversity:.3f} vs {ab.mode_b}={ab.mode_b_avg_diversity:.3f} -> Winner: {ab.winner_diversity}")
    print(f"  Relevance: {ab.mode_a}={ab.mode_a_avg_relevance:.3f} vs {ab.mode_b}={ab.mode_b_avg_relevance:.3f} -> Winner: {ab.winner_relevance}")
    print(f"  Overall winner: {ab.winner_overall}")
