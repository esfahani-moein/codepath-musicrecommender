"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from src.recommender import load_songs, recommend_songs, format_recommendations_table, SCORING_MODES


# Define diverse user profiles for evaluation
PROFILES = {
    "High-Energy Pop": {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.85,
        "valence": 0.8,
        "danceability": 0.8,
        "likes_acoustic": False,
    },
    "Chill Lofi": {
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.35,
        "valence": 0.6,
        "danceability": 0.55,
        "likes_acoustic": True,
    },
    "Deep Intense Rock": {
        "genre": "rock",
        "mood": "intense",
        "energy": 0.9,
        "valence": 0.45,
        "danceability": 0.6,
        "likes_acoustic": False,
    },
    "Conflicting Profile (High Energy + Sad)": {
        "genre": "lofi",
        "mood": "intense",
        "energy": 0.9,
        "valence": 0.3,
        "danceability": 0.7,
        "likes_acoustic": False,
    },
    "Acoustic Jazz Lover": {
        "genre": "jazz",
        "mood": "relaxed",
        "energy": 0.4,
        "valence": 0.7,
        "danceability": 0.5,
        "likes_acoustic": True,
    },
}


def print_recommendations(profile_name: str, user_prefs: dict, songs: list, k: int = 5, mode: str = "balanced", diversity: bool = False) -> None:
    """Print formatted recommendations for a given user profile."""
    sep = "=" * 80
    print(sep)
    print(f"  Profile: {profile_name}")
    print(f"  Prefs: genre={user_prefs['genre']}, mood={user_prefs['mood']}, energy={user_prefs['energy']}")
    print(f"  Mode: {mode} | Diversity penalty: {diversity}")
    print(sep)

    recommendations = recommend_songs(user_prefs, songs, k=k, mode=mode, diversity_penalty=diversity)
    print(format_recommendations_table(recommendations))
    print()


def main() -> None:
    songs = load_songs("data/songs.csv")

    print("\n" + "=" * 80)
    print("  VibeFinder 1.0 - Music Recommender Simulation")
    print("=" * 80)

    # Default profile with balanced mode
    user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8, "valence": 0.8, "danceability": 0.8, "likes_acoustic": False}
    print_recommendations("Default Pop/Happy (Balanced)", user_prefs, songs, mode="balanced")

    # Run all stress-test profiles with balanced mode
    for name, prefs in PROFILES.items():
        print_recommendations(name, prefs, songs, mode="balanced")

    # Demo: Compare scoring modes for the same profile
    print("\n" + "=" * 80)
    print("  SCORING MODE COMPARISON (High-Energy Pop profile)")
    print("=" * 80)
    pop_prefs = PROFILES["High-Energy Pop"]
    for mode_name in SCORING_MODES:
        print_recommendations(f"High-Energy Pop [{mode_name}]", pop_prefs, songs, k=3, mode=mode_name)

    # Demo: Diversity penalty for Chill Lofi
    print("\n" + "=" * 80)
    print("  DIVERSITY PENALTY DEMO (Chill Lofi profile)")
    print("=" * 80)
    lofi_prefs = PROFILES["Chill Lofi"]
    print_recommendations("Chill Lofi [balanced, no diversity]", lofi_prefs, songs, mode="balanced", diversity=False)
    print_recommendations("Chill Lofi [balanced, WITH diversity]", lofi_prefs, songs, mode="balanced", diversity=True)


if __name__ == "__main__":
    main()
