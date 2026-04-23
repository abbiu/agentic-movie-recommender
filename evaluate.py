"""
Run a small manual evaluation suite for the movie recommender.

Usage:
    OLLAMA_API_KEY=your_key_here python evaluate.py
"""

import csv
import os
import re

from llm import CANDIDATE_BY_ID, _fetch_tmdb_movie_details, get_recommendation


TEST_CASES = [
    {
        "id": 1,
        "preferences": "I want a funny, feel-good movie.",
        "history": [],
        "history_ids": [],
        "intent": "Comedy / uplifting",
        "must_avoid": "Dark or horror-heavy picks",
    },
    {
        "id": 2,
        "preferences": "Something dark and suspenseful, but not horror.",
        "history": ["The Dark Knight", "Se7en"],
        "history_ids": [],
        "intent": "Thriller / tension",
        "must_avoid": "Explicit horror",
    },
    {
        "id": 3,
        "preferences": "Give me a romantic movie, but not cheesy.",
        "history": [],
        "history_ids": [],
        "intent": "Romance / mature tone",
        "must_avoid": "Overly sugary rom-com energy",
    },
    {
        "id": 4,
        "preferences": "I want a recent sci-fi movie that feels serious and thoughtful.",
        "history": [],
        "history_ids": [],
        "intent": "Recent serious sci-fi",
        "must_avoid": "Silly / lightweight picks",
    },
    {
        "id": 5,
        "preferences": "A short comfort watch for a stressful night.",
        "history": [],
        "history_ids": [],
        "intent": "Short comfort movie",
        "must_avoid": "Long or emotionally punishing films",
    },
    {
        "id": 6,
        "preferences": "An action movie, but not superheroes.",
        "history": ["The Avengers", "Iron Man"],
        "history_ids": [24428, 1726],
        "intent": "Non-superhero action",
        "must_avoid": "Superhero movies / watched titles",
    },
    {
        "id": 7,
        "preferences": "I want something animated and funny.",
        "history": [],
        "history_ids": [],
        "intent": "Animation / humor",
        "must_avoid": "Non-animated adult drama",
    },
    {
        "id": 8,
        "preferences": "Recommend an older classic movie.",
        "history": [],
        "history_ids": [],
        "intent": "Older / classic",
        "must_avoid": "Very recent releases",
    },
    {
        "id": 9,
        "preferences": "I want a scary movie.",
        "history": [],
        "history_ids": [],
        "intent": "Horror / scary",
        "must_avoid": "Non-scary general thrillers",
    },
    {
        "id": 10,
        "preferences": "I want something light and recent, not dark.",
        "history": [],
        "history_ids": [],
        "intent": "Recent / light tone",
        "must_avoid": "Dark, bleak movies",
    },
    {
        "id": 11,
        "preferences": "I want to watch a movie focusing on world war 2 conflicts with a heavy focus on action scenes",
        "history": ["The Longest Day", "Bridge Over the River Kwai"],
        "history_ids": [],
        "intent": "WWII / historical war action",
        "must_avoid": "Superhero or sci-fi war titles",
    },
    {
        "id": 12,
        "preferences": "animation new release",
        "history": ["mario"],
        "history_ids": [],
        "intent": "Recent animation",
        "must_avoid": "Older animated titles beating newer strong options",
    },
    {
        "id": 13,
        "preferences": "high-rated sci-fi, under 2 hours",
        "history": [],
        "history_ids": [],
        "intent": "High-quality short sci-fi",
        "must_avoid": "Long runtimes or mediocre-quality picks",
    },
    {
        "id": 14,
        "preferences": "romantic movie for couple, not too sad",
        "history": ["Titanic"],
        "history_ids": [],
        "intent": "Date-night romance / warm tone",
        "must_avoid": "Tragic heartbreak-heavy romance",
    },
    {
        "id": 15,
        "preferences": "love story, fun, musical, for couple",
        "history": ["La La Land", "Sing"],
        "history_ids": [],
        "intent": "Musical romance / performance energy",
        "must_avoid": "Generic romance without music/performance feel",
    },
    {
        "id": 16,
        "preferences": "family fantasy but not childish",
        "history": ["Toy Story", "Frozen"],
        "history_ids": [],
        "intent": "Family fantasy with some maturity",
        "must_avoid": "Overly kiddy or already watched picks",
    },
    {
        "id": 17,
        "preferences": "thriller but not too scary",
        "history": [],
        "history_ids": [],
        "intent": "Suspense without horror intensity",
        "must_avoid": "Horror-heavy, gore, or terrifying tone",
    },
    {
        "id": 18,
        "preferences": "comfort movie after work",
        "history": [],
        "history_ids": [],
        "intent": "Warm, low-stress comfort watch",
        "must_avoid": "Intense, punishing, or very dark picks",
    },
    {
        "id": 19,
        "preferences": "action, adventure that friendly for family watch",
        "history": [],
        "history_ids": [],
        "intent": "Family action-adventure",
        "must_avoid": "Hard-R adult action intensity",
    },
    {
        "id": 20,
        "preferences": "i love superheroes and feel-good buddy cop stories",
        "history": ["iron man 3"],
        "history_ids": [],
        "intent": "Superhero team/duo fun",
        "must_avoid": "Random action-comedy without hero/team energy",
    },
    {
        "id": 21,
        "preferences": "new realse sci fi with high imbd rating",
        "history": [],
        "history_ids": [],
        "intent": "Typo-robust recent high-rated sci-fi",
        "must_avoid": "Ignoring typo-normalized rating / recentness signals",
    },
    {
        "id": 22,
        "preferences": "funny romantic movie",
        "history": [],
        "history_ids": [],
        "intent": "Default English rom-com",
        "must_avoid": "Foreign-language pick without explicit locale request",
    },
    {
        "id": 23,
        "preferences": "funny romantic movie in Korean",
        "history": [],
        "history_ids": [],
        "intent": "Korean romantic comedy",
        "must_avoid": "Non-Korean result",
    },
    {
        "id": 24,
        "preferences": "animation, emotional but uplifting",
        "history": ["Inside Out", "Inside Out 2", "Frozen", "Coco"],
        "history_ids": [],
        "intent": "Fresh uplifting animation",
        "must_avoid": "Watched titles or irrelevant fallback picks",
    },
    {
        "id": 25,
        "preferences": "funny romantic movie",
        "history": [],
        "history_ids": [],
        "intent": "Description quality spot-check",
        "must_avoid": "Database-like or overly long explanation",
    },
    {
        "id": 26,
        "preferences": "i love superheros",
        "history": [],
        "history_ids": [],
        "intent": "Typo-robust superhero preference",
        "must_avoid": "Romance/drama drift caused by misspelling",
    },
]


OUTPUT_CSV = os.path.join(
    os.path.dirname(__file__), "evaluation_results.csv"
)

TRAGIC_TERMS = {
    "tragic",
    "tragedy",
    "tearjerker",
    "grief",
    "bleak",
    "heartbreak",
    "heartbreaking",
    "death",
    "loss",
    "mourning",
    "depressing",
    "sad",
}

LIGHT_TERMS = {
    "feel good",
    "feel-good",
    "warm",
    "heartwarming",
    "uplifting",
    "fun",
    "funny",
    "light",
    "easy",
    "comfort",
    "hopeful",
    "playful",
}

INTENSE_TERMS = {
    "dark",
    "bleak",
    "disturbing",
    "gore",
    "violent",
    "terrifying",
    "intense",
    "grim",
    "slasher",
}

MUSICAL_TERMS = {
    "music",
    "musical",
    "sing",
    "singer",
    "song",
    "songs",
    "dance",
    "dancing",
    "performance",
    "performer",
    "band",
    "concert",
}

TEAM_TERMS = {
    "buddy",
    "duo",
    "team",
    "partner",
    "partners",
    "crew",
    "group",
    "alliance",
}

SUPERHERO_TERMS = {
    "superhero",
    "superheroes",
    "comic",
    "marvel",
    "dc",
    "avengers",
    "hero",
}

HIGH_RATING_MIN = 7.5
MIN_RATING_VOTES = 200


def _contains_any(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def _top_stars_text(movie: dict, limit: int = 2) -> list[str]:
    cast = [
        name.strip()
        for name in str(movie.get("top_cast", "") or "").split(",")
        if name.strip()
    ]
    return cast[:limit]


def quick_eval(case: dict, movie: dict) -> str:
    preferences = case["preferences"].lower()
    genres = movie["genres_text"].lower()
    metadata_text = (
        f"{movie['overview']} {movie['keywords']} {movie['tagline']} "
        f"{movie.get('top_cast', '')} {movie.get('director', '')} "
        f"{movie.get('original_language', '')}"
    ).lower()
    language = str(movie.get("original_language") or "").lower()
    year = int(movie["year"] or 0)
    runtime = int(movie["runtime_min"] or 0)
    rating = float(movie.get("vote_average") or 0.0)

    if "not superheroes" in preferences and "superhero" in metadata_text:
        return "Check: superhero leakage"
    if "animated" in preferences and "animation" not in genres:
        return "Check: missed animation"
    if "scary" in preferences and "horror" not in genres:
        return "Check: scary without horror tag"
    if "older classic" in preferences and year > 2005:
        return "Check: not very classic"
    if "recent" in preferences and year < 2020:
        return "Check: not very recent"
    if "short" in preferences and runtime > 125:
        return "Check: runtime seems long"
    if "not dark" in preferences and (
        "dark" in metadata_text or "thriller" in genres
    ):
        return "Check: may still feel dark"
    if "world war 2" in preferences or "wwii" in preferences or "ww2" in preferences:
        if "war" not in genres and "history" not in genres:
            return "Check: missed WWII/war signal"
        if "superhero" in metadata_text or "science fiction" in genres:
            return "Check: war prompt drifted to franchise/sci-fi"
    if "animation new release" in preferences:
        if "animation" not in genres:
            return "Check: missed animation"
        if year < 2023:
            return "Check: not new enough"
    if "under 2 hours" in preferences:
        if "science fiction" not in genres:
            return "Check: missed sci-fi"
        if runtime > 120:
            return "Check: runtime too long"
        if rating < 7.0:
            return "Check: rating not strong enough"
    if "not too sad" in preferences and _contains_any(metadata_text, TRAGIC_TERMS):
        return "Check: still looks too sad"
    if "musical" in preferences and not _contains_any(metadata_text + " " + genres, MUSICAL_TERMS):
        return "Check: musical signal missed"
    if "family fantasy" in preferences:
        if "fantasy" not in genres:
            return "Check: missed fantasy"
        if "family" not in genres and "animation" not in genres:
            return "Check: not family-friendly enough"
    if "thriller but not too scary" in preferences and "horror" in genres:
        return "Check: too horror-heavy"
    if "comfort movie after work" in preferences:
        if runtime > 130:
            return "Check: long for comfort watch"
        if _contains_any(metadata_text, INTENSE_TERMS):
            return "Check: too intense for comfort watch"
    if "friendly for family watch" in preferences:
        if "action" not in genres and "adventure" not in genres:
            return "Check: missed action-adventure"
        if "family" not in genres and "animation" not in genres:
            return "Check: not family-friendly"
    if "superheroes and feel-good buddy cop stories" in preferences:
        if "superhero" not in metadata_text and "comic" not in metadata_text:
            return "Check: superhero signal weak"
        if not _contains_any(metadata_text, TEAM_TERMS):
            return "Check: team/buddy dynamic weak"
    if "high imbd rating" in preferences:
        if "science fiction" not in genres:
            return "Check: missed sci-fi after typo normalization"
        if year < 2020:
            return "Check: typo case not recent enough"
    if "superheros" in preferences:
        if not _contains_any(metadata_text + " " + genres, SUPERHERO_TERMS):
            return "Check: superhero typo not understood"
    if case["id"] == 22 and language and language != "en":
        return "Check: default English preference not met"
    if "in korean" in preferences and language != "ko":
        return "Check: Korean language preference missed"
    if case["id"] == 24 and (
        movie["title"].strip().casefold() in {title.strip().casefold() for title in case["history"]}
    ):
        return "Check: watched title repeated"
    return "Reasonable fit"


def score_case(case: dict, movie: dict) -> tuple[int, int, int, str]:
    preferences = case["preferences"].lower()
    genres = movie["genres_text"].lower()
    metadata_text = (
        f"{movie['overview']} {movie['keywords']} {movie['tagline']} "
        f"{movie.get('top_cast', '')} {movie.get('director', '')} "
        f"{movie.get('original_language', '')}"
    ).lower()
    language = str(movie.get("original_language") or "").lower()
    title_norm = movie["title"].strip().casefold()
    history_norm = {title.strip().casefold() for title in case["history"]}
    history_id_set = {int(value) for value in case["history_ids"] or []}
    year = int(movie["year"] or 0)
    runtime = int(movie["runtime_min"] or 0)
    rating = float(movie.get("vote_average") or 0.0)

    constraint_score = 1
    notes: list[str] = []

    if "not superheroes" in preferences and "superhero" in metadata_text:
        constraint_score = 0
        notes.append("violates non-superhero constraint")
    elif "not horror" in preferences and "horror" in genres:
        constraint_score = 0
        notes.append("violates non-horror constraint")
    elif "not dark" in preferences and (
        "dark" in metadata_text or "thriller" in genres
    ):
        constraint_score = 0
        notes.append("still looks dark")
    elif "short" in preferences and runtime > 125:
        constraint_score = 0
        notes.append("too long for short-watch request")
    elif "recent" in preferences and year < 2020:
        constraint_score = 0
        notes.append("not recent enough")
    elif "older classic" in preferences and year > 2005:
        constraint_score = 0
        notes.append("not old/classic enough")
    elif ("world war 2" in preferences or "wwii" in preferences or "ww2" in preferences) and (
        "war" not in genres and "history" not in genres
    ):
        constraint_score = 0
        notes.append("does not look like a WWII/war title")
    elif ("world war 2" in preferences or "wwii" in preferences or "ww2" in preferences) and (
        "superhero" in metadata_text or "science fiction" in genres
    ):
        constraint_score = 0
        notes.append("drifted into franchise/sci-fi war wording")
    elif "animation new release" in preferences and year < 2023:
        constraint_score = 0
        notes.append("animation pick is not recent/new enough")
    elif "under 2 hours" in preferences and runtime > 120:
        constraint_score = 0
        notes.append("runtime exceeds 120 minutes")
    elif "not too sad" in preferences and _contains_any(metadata_text, TRAGIC_TERMS):
        constraint_score = 0
        notes.append("romance still looks too tragic/sad")
    elif "musical" in preferences and not _contains_any(metadata_text + " " + genres, MUSICAL_TERMS):
        constraint_score = 0
        notes.append("ignored musical/performance requirement")
    elif "family fantasy" in preferences and "fantasy" not in genres:
        constraint_score = 0
        notes.append("missed fantasy requirement")
    elif "thriller but not too scary" in preferences and "horror" in genres:
        constraint_score = 0
        notes.append("too horror-heavy for low-scare thriller request")
    elif "comfort movie after work" in preferences and _contains_any(metadata_text, INTENSE_TERMS):
        constraint_score = 0
        notes.append("too intense/dark for after-work comfort watch")
    elif "friendly for family watch" in preferences and (
        "family" not in genres and "animation" not in genres
    ):
        constraint_score = 0
        notes.append("not family-friendly enough")
    elif "high imbd rating" in preferences and year < 2020:
        constraint_score = 0
        notes.append("did not honor recentness after typo normalization")
    elif "superheros" in preferences and not _contains_any(
        metadata_text + " " + genres, SUPERHERO_TERMS
    ):
        constraint_score = 0
        notes.append("did not honor superhero intent after typo")
    elif case["id"] == 22 and language and language != "en":
        constraint_score = 0
        notes.append("did not default to English")
    elif "in korean" in preferences and language != "ko":
        constraint_score = 0
        notes.append("did not honor Korean language preference")

    intent_score = 0
    if "funny" in preferences or "feel-good" in preferences:
        if "comedy" in genres or "family" in genres:
            intent_score = 2
        elif "drama" in genres:
            intent_score = 1
    elif "dark and suspenseful" in preferences:
        if "thriller" in genres or "crime" in genres:
            intent_score = 2
        elif "drama" in genres:
            intent_score = 1
    elif "romantic" in preferences:
        if "romance" in genres:
            intent_score = 2
        elif "drama" in genres:
            intent_score = 1
    elif "sci-fi" in preferences or "sci fi" in preferences:
        if "science fiction" in genres:
            intent_score = 2
        elif "adventure" in genres:
            intent_score = 1
    elif "comfort watch" in preferences or "stressful night" in preferences:
        if "family" in genres or "comedy" in genres or "animation" in genres:
            intent_score = 2
        elif "adventure" in genres:
            intent_score = 1
    elif "action movie" in preferences:
        if "action" in genres:
            intent_score = 2
        elif "adventure" in genres:
            intent_score = 1
    elif "animated and funny" in preferences:
        if "animation" in genres and "comedy" in genres:
            intent_score = 2
        elif "animation" in genres:
            intent_score = 1
    elif "older classic" in preferences:
        if year <= 2005:
            intent_score = 2
        elif year <= 2015:
            intent_score = 1
    elif "scary" in preferences:
        if "horror" in genres:
            intent_score = 2
        elif "thriller" in genres:
            intent_score = 1
    elif "light and recent" in preferences:
        if year >= 2020 and (
            "comedy" in genres or "family" in genres or "animation" in genres
        ):
            intent_score = 2
        elif "comedy" in genres or "family" in genres:
            intent_score = 1
    elif "world war 2" in preferences or "wwii" in preferences or "ww2" in preferences:
        if "war" in genres and "action" in genres:
            intent_score = 2
        elif "war" in genres or "history" in genres:
            intent_score = 1
    elif "animation new release" in preferences:
        if "animation" in genres and year >= 2024:
            intent_score = 2
        elif "animation" in genres and year >= 2022:
            intent_score = 1
    elif "under 2 hours" in preferences:
        if "science fiction" in genres and runtime <= 120 and rating >= 7.3:
            intent_score = 2
        elif "science fiction" in genres and runtime <= 120 and rating >= 6.8:
            intent_score = 1
    elif "not too sad" in preferences:
        if "romance" in genres and not _contains_any(metadata_text, TRAGIC_TERMS):
            intent_score = 2
        elif "romance" in genres:
            intent_score = 1
    elif "musical" in preferences:
        if "romance" in genres and _contains_any(metadata_text + " " + genres, MUSICAL_TERMS):
            intent_score = 2
        elif _contains_any(metadata_text + " " + genres, MUSICAL_TERMS):
            intent_score = 1
    elif "family fantasy" in preferences:
        if "fantasy" in genres and ("family" in genres or "animation" in genres):
            intent_score = 2
        elif "fantasy" in genres:
            intent_score = 1
    elif "thriller but not too scary" in preferences:
        if "thriller" in genres and "horror" not in genres:
            intent_score = 2
        elif "mystery" in genres and "horror" not in genres:
            intent_score = 1
    elif "comfort movie after work" in preferences:
        if (
            runtime <= 125
            and ("comedy" in genres or "family" in genres or "animation" in genres)
            and not _contains_any(metadata_text, INTENSE_TERMS)
        ):
            intent_score = 2
        elif runtime <= 130 and not _contains_any(metadata_text, INTENSE_TERMS):
            intent_score = 1
    elif "friendly for family watch" in preferences:
        if ("action" in genres or "adventure" in genres) and (
            "family" in genres or "animation" in genres
        ):
            intent_score = 2
        elif "adventure" in genres:
            intent_score = 1
    elif "superheroes and feel-good buddy cop stories" in preferences:
        if (
            ("superhero" in metadata_text or "comic" in metadata_text)
            and _contains_any(metadata_text, TEAM_TERMS)
        ):
            intent_score = 2
        elif "superhero" in metadata_text or "comic" in metadata_text:
            intent_score = 1
    elif "high imbd rating" in preferences:
        if "science fiction" in genres and year >= 2020 and rating >= 7.3:
            intent_score = 2
        elif "science fiction" in genres and rating >= 6.8:
            intent_score = 1
    elif case["id"] == 22:
        if language == "en" and "romance" in genres and ("comedy" in genres or "family" in genres):
            intent_score = 2
        elif language == "en" and "romance" in genres:
            intent_score = 1
    elif "in korean" in preferences:
        if language == "ko" and "romance" in genres and (
            "comedy" in genres or "drama" in genres
        ):
            intent_score = 2
        elif language == "ko":
            intent_score = 1
    elif case["id"] == 24:
        if (
            "animation" in genres
            and not _contains_any(metadata_text, TRAGIC_TERMS)
            and title_norm not in history_norm
        ):
            intent_score = 2
        elif "animation" in genres and title_norm not in history_norm:
            intent_score = 1
    elif "superheros" in preferences:
        if _contains_any(metadata_text + " " + genres, SUPERHERO_TERMS):
            intent_score = 2
        elif "action" in genres or "science fiction" in genres:
            intent_score = 1

    history_score = 1
    if movie["tmdb_id"] in history_id_set or title_norm in history_norm:
        history_score = 0
        notes.append("recommends watched title")

    note_text = "; ".join(notes) if notes else "passes heuristic checks"
    return (
        constraint_score,
        intent_score,
        history_score,
        note_text,
    )


def score_tmdb_metadata(tmdb_details: dict | None) -> tuple[int, str]:
    if not tmdb_details:
        return 0, "TMDB enrichment unavailable"

    fields = {
        "overview": bool(tmdb_details.get("overview")),
        "tagline": bool(tmdb_details.get("tagline")),
        "runtime": bool(tmdb_details.get("runtime_min")),
        "director": bool(tmdb_details.get("director")),
        "top_cast": bool(tmdb_details.get("top_cast")),
        "keywords": bool(tmdb_details.get("keywords")),
    }
    score = sum(1 for available in fields.values() if available)
    missing = [name for name, available in fields.items() if not available]
    note = "all key TMDB fields available" if not missing else f"missing: {', '.join(missing)}"
    return score, note


def score_description(description: str, movie: dict, tmdb_details: dict | None) -> tuple[int, str]:
    description_text = description.lower()
    if not description_text:
        return 0, "empty description"

    metadata = tmdb_details or movie
    evidence_hits = 0
    evidence: list[str] = []

    director = str(metadata.get("director") or "").strip()
    if director and director.lower() in description_text:
        evidence_hits += 1
        evidence.append("director")

    tagline = str(metadata.get("tagline") or "").strip()
    if tagline:
        tagline_tokens = {
            token for token in re.findall(r"[a-z0-9]+", tagline.lower()) if len(token) > 4
        }
        if tagline_tokens and any(token in description_text for token in tagline_tokens):
            evidence_hits += 1
            evidence.append("tagline")

    cast = [
        name.strip()
        for name in str(metadata.get("top_cast") or "").split(",")
        if name.strip()
    ]
    if any(name.lower() in description_text for name in cast[:3]):
        evidence_hits += 1
        evidence.append("cast")

    keywords = [
        keyword.strip()
        for keyword in str(metadata.get("keywords") or "").split(",")
        if keyword.strip()
    ]
    if any(keyword.lower() in description_text for keyword in keywords[:12]):
        evidence_hits += 1
        evidence.append("keywords")

    overview = str(metadata.get("overview") or "")
    overview_tokens = {
        token for token in re.findall(r"[a-z0-9]+", overview.lower()) if len(token) > 6
    }
    if overview_tokens and sum(1 for token in overview_tokens if token in description_text) >= 2:
        evidence_hits += 1
        evidence.append("overview")

    if evidence_hits >= 2:
        return 2, f"uses metadata: {', '.join(evidence)}"
    if evidence_hits == 1:
        return 1, f"uses some metadata: {', '.join(evidence)}"
    return 0, "description does not visibly use TMDB metadata"


def score_description_quality(case: dict, description: str) -> tuple[int, str]:
    description_text = re.sub(r"\s+", " ", str(description or "")).strip()
    if not description_text:
        return 0, "empty description"

    score = 0
    notes: list[str] = []
    lowered = description_text.lower()
    preference_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", case["preferences"].lower())
        if len(token) > 3 and token not in {"movie", "watch", "stories", "story"}
    }

    if len(description_text) <= 500:
        score += 1
        notes.append("under 500 chars")
    else:
        notes.append("too long")

    if any(token in lowered for token in preference_tokens):
        score += 1
        notes.append("references prompt fit")
    else:
        notes.append("fit rationale is generic")

    if any(phrase in lowered for phrase in ("perfect pick", "great choice", "fits", "if you're", "you'll get")):
        score += 1
        notes.append("sounds recommendation-like")
    else:
        notes.append("reads more like plain metadata")

    return score, "; ".join(notes)


def score_blurb_format(movie: dict, description: str) -> tuple[int, str]:
    description_text = re.sub(r"\s+", " ", str(description or "")).strip()
    if not description_text:
        return 0, "empty description"

    score = 0
    notes: list[str] = []
    lowered = description_text.lower()

    if str(movie["title"]).lower() in lowered:
        score += 1
        notes.append("includes title")
    else:
        notes.append("missing title")

    year = str(int(movie["year"] or 0)) if movie.get("year") else ""
    if year and year in description_text:
        score += 1
        notes.append("includes year")
    else:
        notes.append("missing year")

    runtime = int(movie.get("runtime_min") or 0)
    if runtime and (f"{runtime} min" in lowered or f"{runtime}min" in lowered):
        score += 1
        notes.append("includes runtime")
    else:
        notes.append("missing runtime")

    stars = _top_stars_text(movie)
    star_hits = sum(1 for star in stars if star.lower() in lowered)
    if star_hits >= min(2, len(stars)) and stars:
        score += 1
        notes.append("includes top stars")
    elif star_hits >= 1 and stars:
        notes.append("includes partial stars")
    else:
        notes.append("missing stars")

    vote_average = float(movie.get("vote_average") or 0.0)
    vote_count = int(float(movie.get("vote_count") or 0.0))
    rating_required = vote_average >= HIGH_RATING_MIN and vote_count > MIN_RATING_VOTES
    if rating_required:
        if "highly rated" in lowered or "/10" in lowered:
            score += 1
            notes.append("includes rating note")
        else:
            notes.append("missing rating note")
    else:
        score += 1
        notes.append("rating note not required")

    return score, "; ".join(notes)


def main() -> None:
    rows: list[dict[str, object]] = []
    print(
        "| # | Preferences | History | Intent | Must Avoid | Result | tmdb_id | Quick Eval |"
    )
    print("|---|---|---|---|---|---|---:|---|")

    for case in TEST_CASES:
        result = get_recommendation(
            case["preferences"], case["history"], case["history_ids"]
        )
        movie = CANDIDATE_BY_ID[result["tmdb_id"]]
        history_text = ", ".join(case["history"]) if case["history"] else "-"
        result_text = (
            f"{movie['title']} ({movie['year']}) [{movie['genres_text']}]"
        )
        assessment = quick_eval(case, movie)
        (
            constraint_score,
            intent_score,
            history_score,
            notes,
        ) = score_case(case, movie)
        tmdb_details = _fetch_tmdb_movie_details(int(result["tmdb_id"]))
        tmdb_enriched = tmdb_details is not None
        metadata_score, metadata_notes = score_tmdb_metadata(tmdb_details)
        description_score, description_notes = score_description(
            str(result.get("description", "")), movie, tmdb_details
        )
        quality_score, quality_notes = score_description_quality(
            case, str(result.get("description", ""))
        )
        blurb_format_score, blurb_format_notes = score_blurb_format(
            movie, str(result.get("description", ""))
        )
        overall_score = (
            constraint_score
            + intent_score
            + history_score
            + metadata_score
            + description_score
            + quality_score
            + blurb_format_score
        )
        print(
            f"| {case['id']} | {case['preferences']} | {history_text} | "
            f"{case['intent']} | {case['must_avoid']} | {result_text} | "
            f"{result['tmdb_id']} | {assessment} |"
        )
        rows.append(
            {
                "case_id": case["id"],
                "preferences": case["preferences"],
                "history": history_text,
                "intent": case["intent"],
                "must_avoid": case["must_avoid"],
                "recommended_title": movie["title"],
                "year": movie["year"],
                "genres": movie["genres_text"],
                "tmdb_id": result["tmdb_id"],
                "description": result.get("description", ""),
                "tmdb_enriched": tmdb_enriched,
                "tmdb_runtime_min": (tmdb_details or {}).get("runtime_min", ""),
                "tmdb_director": (tmdb_details or {}).get("director", ""),
                "tmdb_top_cast": (tmdb_details or {}).get("top_cast", ""),
                "tmdb_tagline": (tmdb_details or {}).get("tagline", ""),
                "tmdb_keywords": (tmdb_details or {}).get("keywords", ""),
                "constraint_score": constraint_score,
                "intent_score": intent_score,
                "history_score": history_score,
                "tmdb_metadata_score": metadata_score,
                "description_score": description_score,
                "description_quality_score": quality_score,
                "blurb_format_score": blurb_format_score,
                "overall_score": overall_score,
                "quick_eval": assessment,
                "notes": notes,
                "tmdb_notes": metadata_notes,
                "description_notes": description_notes,
                "quality_notes": quality_notes,
                "blurb_format_notes": blurb_format_notes,
            }
        )

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "case_id",
                "preferences",
                "history",
                "intent",
                "must_avoid",
                "recommended_title",
                "year",
                "genres",
                "tmdb_id",
                "description",
                "tmdb_enriched",
                "tmdb_runtime_min",
                "tmdb_director",
                "tmdb_top_cast",
                "tmdb_tagline",
                "tmdb_keywords",
                "constraint_score",
                "intent_score",
                "history_score",
                "tmdb_metadata_score",
                "description_score",
                "description_quality_score",
                "blurb_format_score",
                "overall_score",
                "quick_eval",
                "notes",
                "tmdb_notes",
                "description_notes",
                "quality_notes",
                "blurb_format_notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
