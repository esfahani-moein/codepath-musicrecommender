"""
VibeFinder 2.0 - Enhanced Music Recommender with RAG Integration.

This module extends the original recommender with:
- RAG-augmented scoring (related genres, mood guidance from knowledge base)
- Fuzzy genre matching (indie pop partially matches pop)
- Song knowledge context in explanations
- Diversity and novelty penalties
- Multiple scoring modes with RAG enrichment
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

from src.rag_engine import MusicKnowledgeBase, RAGEnrichedRecommender


@dataclass
class Song:
    """Represents a song and its attributes."""
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    """Represents a user's taste preferences."""
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    target_valence: Optional[float] = None
    target_danceability: Optional[float] = None


class Recommender:
    """OOP implementation of the recommendation logic."""

    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5, rag_enricher: Optional[RAGEnrichedRecommender] = None) -> List[Song]:
        """Return top-k songs ranked by score for the given user."""
        user_prefs = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
        }
        if user.target_valence is not None:
            user_prefs["valence"] = user.target_valence
        if user.target_danceability is not None:
            user_prefs["danceability"] = user.target_danceability

        songs_dict = [self._song_to_dict(s) for s in self.songs]
        results = recommend_songs(user_prefs, songs_dict, k=k, rag_enricher=rag_enricher)
        # Return original Song objects in ranked order
        song_ids = {s.id: s for s in self.songs}
        ranked = []
        for song_dict, score, explanation in results:
            ranked.append(song_ids[song_dict["id"]])
        return ranked

    def explain_recommendation(self, user: UserProfile, song: Song, rag_enricher: Optional[RAGEnrichedRecommender] = None) -> str:
        """Return a human-readable explanation for why a song was recommended."""
        user_prefs = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
        }
        if user.target_valence is not None:
            user_prefs["valence"] = user.target_valence
        if user.target_danceability is not None:
            user_prefs["danceability"] = user.target_danceability

        song_dict = self._song_to_dict(song)
        score, reasons = score_song_rag(user_prefs, song_dict, rag_enricher=rag_enricher)

        if rag_enricher is not None:
            return rag_enricher.generate_explanation_with_rag(user_prefs, song_dict, score, reasons)
        return "; ".join(reasons)

    @staticmethod
    def _song_to_dict(song: Song) -> Dict[str, Any]:
        return {
            "id": song.id,
            "title": song.title,
            "artist": song.artist,
            "genre": song.genre,
            "mood": song.mood,
            "energy": song.energy,
            "tempo_bpm": song.tempo_bpm,
            "valence": song.valence,
            "danceability": song.danceability,
            "acousticness": song.acousticness,
        }


SCORING_MODES = {
    "balanced": {"genre": 2.0, "mood": 1.0, "energy": 1.5, "valence": 0.5, "danceability": 0.5, "acoustic": 0.5, "related_genre": 1.0},
    "genre_first": {"genre": 3.0, "mood": 0.5, "energy": 1.0, "valence": 0.3, "danceability": 0.3, "acoustic": 0.3, "related_genre": 1.5},
    "mood_first": {"genre": 0.5, "mood": 3.0, "energy": 1.0, "valence": 0.5, "danceability": 0.5, "acoustic": 0.3, "related_genre": 0.3},
    "energy_focused": {"genre": 1.0, "mood": 0.5, "energy": 3.0, "valence": 0.5, "danceability": 0.5, "acoustic": 0.3, "related_genre": 0.5},
    "rag_enhanced": {"genre": 1.5, "mood": 1.5, "energy": 1.5, "valence": 0.5, "danceability": 0.5, "acoustic": 0.5, "related_genre": 1.5},
}


def load_songs(csv_path: str) -> List[Dict[str, Any]]:
    """Loads songs from a CSV file and converts numerical fields to floats."""
    songs = []
    float_fields = {"energy", "tempo_bpm", "valence", "danceability", "acousticness"}
    int_fields = {"id"}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            song = {}
            for key, val in row.items():
                key = key.strip()
                val = val.strip()
                if key in int_fields:
                    song[key] = int(val)
                elif key in float_fields:
                    song[key] = float(val)
                else:
                    song[key] = val
            songs.append(song)
    return songs


def _genre_match_score(user_prefs: Dict, song: Dict, weights: Dict, rag_enricher: Optional[RAGEnrichedRecommender]) -> Tuple[float, List[str]]:
    """Compute genre match score with exact and related genre matching."""
    score = 0.0
    reasons = []
    user_genre = user_prefs.get("genre", "").lower()
    song_genre = song.get("genre", "").lower()

    if user_genre == song_genre:
        pts = weights["genre"]
        score += pts
        reasons.append(f"genre match (+{pts:.1f})")
        return score, reasons

    # Related genre matching via RAG
    related_genres = []
    if rag_enricher is not None and user_genre:
        related_genres = rag_enricher.kb.get_related_genres(user_genre)

    if song_genre in related_genres:
        pts = weights.get("related_genre", 1.0)
        score += pts
        reasons.append(f"related genre match ({song_genre} ~ {user_genre}) (+{pts:.1f})")

    return score, reasons


def score_song_rag(user_prefs: Dict, song: Dict, mode: str = "balanced", rag_enricher: Optional[RAGEnrichedRecommender] = None) -> Tuple[float, List[str]]:
    """Scores a single song using RAG-augmented weighted rules."""
    weights = SCORING_MODES.get(mode, SCORING_MODES["balanced"])
    score = 0.0
    reasons = []

    # Genre match (exact + related)
    g_score, g_reasons = _genre_match_score(user_prefs, song, weights, rag_enricher)
    score += g_score
    reasons.extend(g_reasons)

    # Mood match
    if user_prefs.get("mood", "").lower() == song.get("mood", "").lower():
        pts = weights["mood"]
        score += pts
        reasons.append(f"mood match (+{pts:.1f})")

    # Energy similarity
    target_energy = user_prefs.get("energy", 0.5)
    song_energy = song.get("energy", 0.5)
    energy_sim = (1.0 - abs(song_energy - target_energy)) * weights["energy"]
    score += energy_sim
    reasons.append(f"energy similarity (+{energy_sim:.2f})")

    # Valence similarity
    target_valence = user_prefs.get("valence", 0.5)
    song_valence = song.get("valence", 0.5)
    valence_sim = (1.0 - abs(song_valence - target_valence)) * weights["valence"]
    score += valence_sim
    reasons.append(f"valence similarity (+{valence_sim:.2f})")

    # Danceability similarity
    target_dance = user_prefs.get("danceability", 0.5)
    song_dance = song.get("danceability", 0.5)
    dance_sim = (1.0 - abs(song_dance - target_dance)) * weights["danceability"]
    score += dance_sim
    reasons.append(f"danceability similarity (+{dance_sim:.2f})")

    # Acoustic bonus
    if user_prefs.get("likes_acoustic", False) and song.get("acousticness", 0.0) > 0.5:
        score += weights["acoustic"]
        reasons.append(f"acoustic bonus (+{weights['acoustic']:.1f})")

    # RAG mood guidance bonus
    if rag_enricher is not None:
        mood = user_prefs.get("mood", "").lower()
        if mood:
            guidance = rag_enricher.kb.get_mood_recommendations(mood)
            rec_genres = [g.lower() for g in guidance.get("recommended_genres", [])]
            if song.get("genre", "").lower() in rec_genres and g_score == 0:
                bonus = 0.3
                score += bonus
                reasons.append(f"RAG mood-genre alignment (+{bonus:.1f})")

    return score, reasons


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5, mode: str = "balanced",
                    diversity_penalty: bool = False, novelty_boost: bool = False,
                    rag_enricher: Optional[RAGEnrichedRecommender] = None) -> List[Tuple[Dict, float, str]]:
    """Score all songs and return top-k sorted by score descending."""
    scored = []
    for song in songs:
        score, reasons = score_song_rag(user_prefs, song, mode, rag_enricher)
        explanation = "; ".join(reasons)
        scored.append((song, score, explanation))
    scored.sort(key=lambda x: x[1], reverse=True)

    if diversity_penalty or novelty_boost:
        selected = []
        selected_artists = set()
        selected_genres = set()
        for song, score, explanation in scored:
            penalty = 0.0
            bonus = 0.0
            if diversity_penalty:
                if song["artist"] in selected_artists:
                    penalty += 1.0
                if song["genre"] in selected_genres:
                    penalty += 0.5
            if novelty_boost:
                # Boost songs with unusual genre/artist combos
                if song["genre"] not in selected_genres and song["artist"] not in selected_artists:
                    bonus += 0.2
            adjusted_score = score - penalty + bonus
            if penalty > 0:
                explanation += f"; diversity penalty (-{penalty:.1f})"
            if bonus > 0:
                explanation += f"; novelty boost (+{bonus:.1f})"
            selected.append((song, adjusted_score, explanation))
            selected_artists.add(song["artist"])
            selected_genres.add(song["genre"])
        selected.sort(key=lambda x: x[1], reverse=True)
        return selected[:k]

    return scored[:k]


def format_recommendations_table(recommendations: List[Tuple[Dict, float, str]]) -> str:
    """Format recommendations as a readable ASCII table."""
    if not recommendations:
        return "No recommendations found."

    header = f"{'#':<3} {'Title':<25} {'Artist':<18} {'Genre':<12} {'Score':>6}  {'Reasons'}"
    separator = "-" * len(header)
    lines = [header, separator]

    for i, (song, score, explanation) in enumerate(recommendations, 1):
        title = song.get("title", "?")[:24]
        artist = song.get("artist", "?")[:17]
        genre = song.get("genre", "?")[:11]
        lines.append(f"{i:<3} {title:<25} {artist:<18} {genre:<12} {score:>6.2f}  {explanation}")

    return "\n".join(lines)


def get_rich_explanation(song: Dict, rag_enricher: RAGEnrichedRecommender) -> str:
    """Get a rich textual explanation of a song using the knowledge base."""
    return rag_enricher.kb.get_song_context(song)
