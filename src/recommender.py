import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
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
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return top-k songs ranked by score for the given user."""
        scored = []
        for song in self.songs:
            prefs = {
                "genre": user.favorite_genre,
                "mood": user.favorite_mood,
                "energy": user.target_energy,
                "likes_acoustic": user.likes_acoustic,
            }
            song_dict = {
                "genre": song.genre,
                "mood": song.mood,
                "energy": song.energy,
                "valence": song.valence,
                "danceability": song.danceability,
                "acousticness": song.acousticness,
            }
            score, reasons = score_song(prefs, song_dict)
            scored.append((score, song, reasons))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [song for _, song, _ in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a human-readable explanation for why a song was recommended."""
        prefs = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
        }
        song_dict = {
            "genre": song.genre,
            "mood": song.mood,
            "energy": song.energy,
            "valence": song.valence,
            "danceability": song.danceability,
            "acousticness": song.acousticness,
        }
        _, reasons = score_song(prefs, song_dict)
        return "; ".join(reasons)

def load_songs(csv_path: str) -> List[Dict]:
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
    print(f"Loaded {len(songs)} songs from {csv_path}")
    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Scores a single song against user preferences using weighted rules."""
    score = 0.0
    reasons = []

    # Genre match: +2.0 points
    if user_prefs.get("genre", "").lower() == song.get("genre", "").lower():
        score += 2.0
        reasons.append("genre match (+2.0)")

    # Mood match: +1.0 points
    if user_prefs.get("mood", "").lower() == song.get("mood", "").lower():
        score += 1.0
        reasons.append("mood match (+1.0)")

    # Energy similarity: (1 - |diff|) * 1.5, max 1.5 points
    target_energy = user_prefs.get("energy", 0.5)
    song_energy = song.get("energy", 0.5)
    energy_sim = (1.0 - abs(song_energy - target_energy)) * 1.5
    score += energy_sim
    reasons.append(f"energy similarity (+{energy_sim:.2f})")

    # Valence similarity: (1 - |diff|) * 0.5
    target_valence = user_prefs.get("valence", 0.5)
    song_valence = song.get("valence", 0.5)
    valence_sim = (1.0 - abs(song_valence - target_valence)) * 0.5
    score += valence_sim
    reasons.append(f"valence similarity (+{valence_sim:.2f})")

    # Danceability similarity: (1 - |diff|) * 0.5
    target_dance = user_prefs.get("danceability", 0.5)
    song_dance = song.get("danceability", 0.5)
    dance_sim = (1.0 - abs(song_dance - target_dance)) * 0.5
    score += dance_sim
    reasons.append(f"danceability similarity (+{dance_sim:.2f})")

    # Acousticness bonus: +0.5 if user likes acoustic and song is acoustic
    if user_prefs.get("likes_acoustic", False) and song.get("acousticness", 0.0) > 0.5:
        score += 0.5
        reasons.append("acoustic bonus (+0.5)")

    return (score, reasons)

SCORING_MODES = {
    "balanced": {"genre": 2.0, "mood": 1.0, "energy": 1.5, "valence": 0.5, "danceability": 0.5, "acoustic": 0.5},
    "genre_first": {"genre": 3.0, "mood": 0.5, "energy": 1.0, "valence": 0.3, "danceability": 0.3, "acoustic": 0.3},
    "mood_first": {"genre": 0.5, "mood": 3.0, "energy": 1.0, "valence": 0.5, "danceability": 0.5, "acoustic": 0.3},
    "energy_focused": {"genre": 1.0, "mood": 0.5, "energy": 3.0, "valence": 0.5, "danceability": 0.5, "acoustic": 0.3},
}

def score_song_with_mode(user_prefs: Dict, song: Dict, mode: str = "balanced") -> Tuple[float, List[str]]:
    """Scores a song using a named scoring mode for adjustable weight strategies."""
    weights = SCORING_MODES.get(mode, SCORING_MODES["balanced"])
    score = 0.0
    reasons = []

    # Genre match
    if user_prefs.get("genre", "").lower() == song.get("genre", "").lower():
        pts = weights["genre"]
        score += pts
        reasons.append(f"genre match (+{pts:.1f})")

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

    return (score, reasons)


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5, mode: str = "balanced", diversity_penalty: bool = False) -> List[Tuple[Dict, float, str]]:
    """Score all songs and return top-k sorted by score descending."""
    scored = []
    for song in songs:
        score, reasons = score_song_with_mode(user_prefs, song, mode)
        explanation = "; ".join(reasons)
        scored.append((song, score, explanation))
    scored.sort(key=lambda x: x[1], reverse=True)

    if diversity_penalty:
        selected = []
        selected_artists = set()
        selected_genres = set()
        for song, score, explanation in scored:
            penalty = 0.0
            if song["artist"] in selected_artists:
                penalty += 1.0
            if song["genre"] in selected_genres:
                penalty += 0.5
            adjusted_score = score - penalty
            if penalty > 0:
                explanation += f"; diversity penalty (-{penalty:.1f})"
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
