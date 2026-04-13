# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name

**VibeFinder 1.0**

---

## 2. Intended Use

VibeFinder 1.0 suggests 3–5 songs from a small catalog of 18 tracks based on a user's preferred genre, mood, energy level, valence, danceability, and acousticness preference. It is designed for classroom exploration of how content-based recommendation systems work. It should NOT be used to make real music recommendations for actual users — the catalog is too small, the scoring is too simple, and the system has known biases.

---

## 3. How the Model Works

The system compares each song's features to the user's stated preferences and assigns a numeric score:

- **Genre match**: If the song's genre exactly matches the user's favorite, it gets 2.0 points. This is the single largest bonus because genre is the strongest signal of musical taste.
- **Mood match**: If the mood matches exactly, the song gets 1.0 point. Mood is important but less defining than genre.
- **Energy similarity**: The system measures how close the song's energy level is to the user's target on a 0–1 scale. The closer they are, the more points (up to 1.5). This rewards songs that "feel" the right intensity, not just high or low energy.
- **Valence and danceability similarity**: Same closeness approach, but worth up to 0.5 points each. These capture whether a song feels as happy or as danceable as the user wants.
- **Acoustic bonus**: If the user likes acoustic music and the song has high acousticness (>0.5), it gets 0.5 extra points.

After scoring every song, the system sorts them from highest to lowest score and returns the top results.

---

## 4. Data

- **18 songs** in `data/songs.csv` (10 original + 8 added)
- **Genres represented**: pop, lofi (3 songs), rock, ambient, jazz, synthwave, indie pop, R&B, hip-hop, classical, country, metal, reggae, folk, EDM
- **Moods represented**: happy, chill, intense, relaxed, moody, focused, soulful, confident, melancholy, nostalgic, aggressive, uplifting, reflective, euphoric
- **Added songs**: 8 new tracks covering R&B, hip-hop, classical, country, metal, reggae, folk, and EDM to diversify the catalog beyond the original pop/lofi/rock/ambient focus
- **Missing**: No Latin, K-pop, punk, blues, or world music genres. No instrumental-only or spoken word tracks. The dataset reflects a Western, streaming-platform-centric view of music.

---

## 5. Strengths

- **Clear profiles work well**: When a user's preferences are coherent (e.g., "lofi + chill + low energy"), the top results feel intuitive and match what a human would pick.
- **Transparent scoring**: Every recommendation comes with a reason list, so users can understand why a song was suggested. This is a major advantage over black-box systems.
- **Numerical similarity works**: The `(1 - |difference|) × weight` formula correctly rewards proximity rather than raw magnitude — a song with energy 0.82 is better for a target of 0.80 than one at 0.40, even though 0.40 is "lower."
- **Acoustic Jazz and Chill Lofi profiles** produce especially good results because their features align well with multiple songs in the catalog.

---

## 6. Limitations and Bias

- **Genre over-prioritization**: At +2.0 points, a genre match can override poor mood or energy fit. "Gym Hero" (pop, intense, energy=0.93) ranks #2 for a "happy pop" profile despite its mismatched mood, simply because it shares the genre.
- **Exact-match only for categories**: "Indie pop" doesn't match "pop", and "chill" doesn't match "relaxed." Real music taste has gradients — these hard boundaries miss related genres and moods.
- **Catalog imbalance**: Lofi has 3 songs while most genres have 1. This gives lofi-preferring users more options and higher chances of a close match, creating an unintentional bias.
- **No conflict detection**: The "Conflicting Profile" (high energy + lofi) exposed that the system doesn't recognize when preferences contradict each other — it just adds up points from whichever features happen to match.
- **Filter bubble risk**: Because genre is weighted so heavily, users who like one genre will almost never see songs from other genres, even if those songs match their mood and energy perfectly.

---

## 7. Evaluation

We tested the system with 5 distinct user profiles:

1. **High-Energy Pop** — Results felt right: "Sunrise City" (#1) is a perfect pop/happy/high-energy match.
2. **Chill Lofi** — Results felt right: "Library Rain" and "Midnight Coding" are both lofi/chill/low-energy with acoustic bonus.
3. **Deep Intense Rock** — Results felt right: "Storm Runner" (#1) is rock/intense/high-energy.
4. **Conflicting Profile (High Energy + Lofi)** — Results were revealing: lofi genre match dominated, pulling in low-energy songs despite the high energy target. "Storm Runner" appeared at #4 via mood match, showing the system can partially recover but the genre weight distorts results.
5. **Acoustic Jazz Lover** — Results felt right: "Coffee Shop Stories" (#1) is a perfect jazz/relaxed/acoustic match.

**Weight experiment**: Doubling energy weight (1.5→3.0) and halving genre weight (2.0→1.0) caused "Rooftop Lights" (indie pop, happy) to jump above "Gym Hero" (pop, intense) for the pop/happy profile — confirming that genre dominance was masking mood mismatches.

**Automated tests**: 2 pytest tests pass — `test_recommend_returns_songs_sorted_by_score` and `test_explain_recommendation_returns_non_empty_string`.

---

## 8. Future Work

- **Fuzzy genre/mood matching**: Use string similarity or a genre taxonomy so "indie pop" partially matches "pop" and "chill" partially matches "relaxed."
- **Diversity penalty**: Prevent the top-k list from being dominated by one artist or genre — penalize songs that are too similar to already-selected recommendations.
- **Multiple scoring modes**: Let users switch between "Genre-First," "Mood-First," and "Energy-Focused" strategies to see how different weightings change results.
- **Collaborative filtering simulation**: Add a "users who liked this also liked" layer on top of the content-based scoring.
- **Larger and more diverse catalog**: Add more songs per genre to reduce catalog imbalance bias.

---

## 9. Personal Reflection

The biggest learning moment was realizing how much weight assignment shapes recommendations. A simple change from 2.0 to 1.0 on genre completely changed the ranking — and in a real system with millions of users, that kind of parameter choice determines what entire demographics get to see. Using AI tools helped me quickly prototype the scoring logic, but I had to double-check the math: the AI initially suggested rewarding higher energy values rather than proximity, which would have biased the system toward high-energy songs regardless of user preference. The most surprising thing was how "real" the recommendations felt even with this simple algorithm — when "Library Rain" came up as the top pick for the Chill Lofi profile, it genuinely felt like something Spotify might suggest. But the conflicting profile experiment was a sobering reminder: simple systems can't handle nuance, and in production, that means real users with complex tastes get ignored. Human judgment still matters most when deciding what to optimize for and whose experience to prioritize.  
