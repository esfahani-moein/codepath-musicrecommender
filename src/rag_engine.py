"""
Retrieval-Augmented Generation (RAG) Engine for Music Recommendations.

This module implements a TF-IDF based retrieval system over the contents/ knowledge base.
It retrieves relevant music knowledge (genres, moods, artists, suggestions) to enrich
recommendations with contextual information before scoring.
"""

import json
import os
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedDocument:
    """A single retrieved document from the knowledge base."""
    source: str
    doc_type: str
    title: str
    text: str
    metadata: Dict[str, Any]
    score: float


class MusicKnowledgeBase:
    """Loads and indexes the music knowledge base from JSON files in contents/."""

    def __init__(self, contents_dir: str = "contents"):
        self.contents_dir = contents_dir
        self.documents: List[Dict[str, Any]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.doc_vectors: Optional[np.ndarray] = None
        self._load_all()
        self._build_index()

    def _load_all(self) -> None:
        """Load all JSON files from contents directory."""
        files = ["genres.json", "moods.json", "artists.json", "suggestions.json"]
        for fname in files:
            path = os.path.join(self.contents_dir, fname)
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            key = fname.replace(".json", "")
            items = data.get(key, [])
            for item in items:
                text = self._item_to_text(key, item)
                self.documents.append({
                    "source": fname,
                    "doc_type": key,
                    "title": item.get("name", item.get("title", "Untitled")),
                    "text": text,
                    "metadata": item,
                })

    @staticmethod
    def _item_to_text(doc_type: str, item: Dict[str, Any]) -> str:
        """Flatten a JSON item into a searchable text string."""
        parts = []
        name = item.get("name") or item.get("title", "")
        if name:
            parts.append(f"Name: {name}")
        if "description" in item:
            parts.append(item["description"])
        if "genre" in item:
            g = item["genre"]
            if isinstance(g, list):
                parts.append(f"Genres: {', '.join(g)}")
            else:
                parts.append(f"Genre: {g}")
        if "moods" in item:
            parts.append(f"Moods: {', '.join(item['moods'])}")
        if "recommended_genres" in item:
            parts.append(f"Recommended genres: {', '.join(item['recommended_genres'])}")
        if "signature_sound" in item:
            parts.append(f"Sound: {item['signature_sound']}")
        if "keywords" in item:
            parts.append(f"Keywords: {', '.join(item['keywords'])}")
        if "related_genres" in item:
            parts.append(f"Related genres: {', '.join(item['related_genres'])}")
        if "activity_contexts" in item:
            parts.append(f"Activities: {', '.join(item['activity_contexts'])}")
        return "\n".join(parts)

    def _build_index(self) -> None:
        """Build TF-IDF index over all documents."""
        if not self.documents:
            return
        texts = [d["text"] for d in self.documents]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
        )
        self.doc_vectors = self.vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedDocument]:
        """Retrieve top-k documents most similar to the query."""
        if self.vectorizer is None or self.doc_vectors is None or not self.documents:
            return []
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append(RetrievedDocument(
                source=doc["source"],
                doc_type=doc["doc_type"],
                title=doc["title"],
                text=doc["text"],
                metadata=doc["metadata"],
                score=float(similarities[idx]),
            ))
        return results

    def retrieve_by_genre(self, genre: str, top_k: int = 2) -> List[RetrievedDocument]:
        """Retrieve documents explicitly matching or related to a genre."""
        query = f"genre {genre} music style characteristics"
        return self.retrieve(query, top_k=top_k)

    def retrieve_by_mood(self, mood: str, top_k: int = 2) -> List[RetrievedDocument]:
        """Retrieve documents explicitly matching or related to a mood."""
        query = f"mood {mood} feeling emotional state music recommendation"
        return self.retrieve(query, top_k=top_k)

    def retrieve_for_song(self, song: Dict[str, Any], top_k: int = 3) -> List[RetrievedDocument]:
        """Retrieve relevant knowledge for a specific song."""
        parts = [song.get("title", ""), song.get("artist", ""), song.get("genre", ""), song.get("mood", "")]
        query = " ".join([p for p in parts if p])
        return self.retrieve(query, top_k=top_k)

    def get_related_genres(self, genre: str) -> List[str]:
        """Return a list of genres related to the given genre using the knowledge base."""
        docs = self.retrieve_by_genre(genre, top_k=3)
        related = set()
        for doc in docs:
            meta = doc.metadata
            rg = meta.get("related_genres", [])
            if isinstance(rg, list):
                related.update([g.lower() for g in rg])
            # Also consider the retrieved doc's own genre
            g = meta.get("genre")
            if isinstance(g, list):
                related.update([x.lower() for x in g])
            elif isinstance(g, str):
                related.add(g.lower())
        # Clean up: remove the exact genre if present
        related.discard(genre.lower())
        return sorted(list(related))

    def get_mood_recommendations(self, mood: str) -> Dict[str, Any]:
        """Return mood-specific recommendations and metadata."""
        docs = self.retrieve_by_mood(mood, top_k=2)
        if not docs:
            return {}
        best = docs[0]
        meta = best.metadata
        return {
            "recommended_genres": meta.get("recommended_genres", []),
            "energy_range": meta.get("energy_range", ""),
            "valence_range": meta.get("valence_range", ""),
            "activity_contexts": meta.get("activity_contexts", []),
            "acoustic_preference": meta.get("acoustic_preference", "either"),
            "playlist_examples": meta.get("playlist_examples", []),
            "explanation": meta.get("description", ""),
        }

    def get_song_context(self, song: Dict[str, Any]) -> str:
        """Generate a rich textual context for a song using retrieved knowledge."""
        docs = self.retrieve_for_song(song, top_k=3)
        parts = []
        for doc in docs:
            desc = doc.metadata.get("description", "")
            if desc:
                parts.append(f"[{doc.doc_type.upper()}] {doc.title}: {desc}")
        return "\n".join(parts)


class RAGEnrichedRecommender:
    """Recommender that uses RAG to enrich user profiles before scoring."""

    def __init__(self, knowledge_base: Optional[MusicKnowledgeBase] = None):
        self.kb = knowledge_base or MusicKnowledgeBase()

    def enrich_user_prefs(self, user_prefs: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant knowledge and augment user preferences with related genres and mood insights."""
        enriched = dict(user_prefs)
        genre = user_prefs.get("genre", "").lower()
        mood = user_prefs.get("mood", "").lower()

        # Retrieve related genres to enable fuzzy matching
        if genre:
            related = self.kb.get_related_genres(genre)
            enriched["related_genres"] = related

        # Retrieve mood-specific guidance
        if mood:
            mood_info = self.kb.get_mood_recommendations(mood)
            enriched["mood_guidance"] = mood_info
            # If user didn't specify valence/energy, suggest from mood knowledge
            if "energy" not in user_prefs and mood_info.get("energy_range"):
                try:
                    rng = mood_info["energy_range"].split("-")
                    enriched["suggested_energy"] = (float(rng[0]) + float(rng[1])) / 2
                except Exception:
                    pass
            if "valence" not in user_prefs and mood_info.get("valence_range"):
                try:
                    rng = mood_info["valence_range"].split("-")
                    enriched["suggested_valence"] = (float(rng[0]) + float(rng[1])) / 2
                except Exception:
                    pass

        return enriched

    def generate_explanation_with_rag(
        self, user_prefs: Dict[str, Any], song: Dict[str, Any], base_score: float, base_reasons: List[str]
    ) -> str:
        """Generate a richer recommendation explanation using retrieved knowledge."""
        context = self.kb.get_song_context(song)
        lines = [f"Score: {base_score:.2f}"]
        lines.append("Matching factors:")
        for r in base_reasons:
            lines.append(f"  - {r}")

        # Add RAG context snippet if available
        if context:
            lines.append("Why this fits your vibe:")
            # Summarize context in 1-2 sentences
            sentences = context.replace("\n", " ").split(". ")
            summary = ". ".join(sentences[:2]) + ("." if sentences[:2] else "")
            lines.append(f"  {summary}")

        return "\n".join(lines)


if __name__ == "__main__":
    kb = MusicKnowledgeBase("contents")
    print(f"Loaded {len(kb.documents)} documents into knowledge base.")
    print("\n--- RAG Demo: Query 'happy energetic pop workout' ---")
    results = kb.retrieve("happy energetic pop workout", top_k=3)
    for r in results:
        print(f"  [{r.doc_type}] {r.title} (score={r.score:.3f})")
    print("\n--- Related genres to 'pop' ---")
    print(kb.get_related_genres("pop"))
    print("\n--- Mood guidance for 'happy' ---")
    print(kb.get_mood_recommendations("happy"))
