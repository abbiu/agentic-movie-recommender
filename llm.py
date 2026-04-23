"""
Hybrid movie recommendation pipeline.

Python performs candidate preparation, history filtering, lightweight
preference parsing, TF-IDF similarity, and deterministic ranking.
Ollama Cloud is used only to write the final short recommendation blurb.
"""

import argparse
import json
import math
import os
import re
import time
import unicodedata
from collections import Counter, OrderedDict
from urllib import error, parse, request

import ollama
import pandas as pd

MODEL = "gemma4:31b-cloud"
MAX_DESCRIPTION_CHARS = 500
CACHE_MAX_SIZE = 64
TMDB_TIMEOUT_SECONDS = 4.0
HIGH_RATING_MIN = 7.5
MIN_RATING_VOTES = 200

DATA_PATH = os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.csv")
TOP_MOVIES = pd.read_csv(DATA_PATH)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "for",
    "from",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "movie",
    "movies",
    "my",
    "of",
    "on",
    "or",
    "something",
    "that",
    "the",
    "to",
    "want",
    "watch",
    "with",
}

GENRE_TERMS = {
    "Action": {"action", "action-packed", "action packed"},
    "Adventure": {"adventure", "adventurous", "quest"},
    "Animation": {"animated", "animation", "anime", "cartoon"},
    "Comedy": {"comedy", "funny", "humor", "humorous", "laugh"},
    "Crime": {"crime", "criminal", "detective", "gangster", "heist", "mafia"},
    "Drama": {"drama", "dramatic", "character-driven", "character driven"},
    "Family": {"family", "kid-friendly", "kids", "children"},
    "Fantasy": {"fantasy", "magical", "magic"},
    "History": {"historical", "history", "period piece", "world war", "ww2", "wwii"},
    "Horror": {"gory", "horror", "scary", "terrifying"},
    "Mystery": {"detective", "mystery", "twisty", "whodunit"},
    "Romance": {"love story", "rom-com", "romance", "romantic", "romcom"},
    "Science Fiction": {"future", "sci-fi", "sci fi", "science fiction", "space"},
    "Thriller": {"suspense", "suspenseful", "tense", "thriller"},
    "War": {"battlefront", "combat", "military", "war", "warfare", "world war", "ww2", "wwii"},
}

MOOD_TERMS = {
    "funny": {"fun", "funny", "hilarious", "witty"},
    "feel_good": {"comfort", "feel good", "feel-good", "heartwarming", "uplifting"},
    "dark": {"bleak", "dark", "grim"},
    "romantic": {"love story", "romance", "romantic"},
    "scary": {"creepy", "frightening", "scary", "terrifying"},
    "serious": {"grounded", "realistic", "serious"},
    "light": {"easygoing", "light", "lighthearted"},
}

MUSICAL_QUERY_TERMS = {
    "music",
    "musical",
    "performance",
    "sing",
    "singing",
    "dance",
    "dancing",
}

MUSICAL_CANDIDATE_TERMS = {
    "music",
    "musical",
    "performance",
    "performer",
    "stage",
    "concert",
    "jazz",
    "sing",
    "singing",
    "singer",
    "dance",
    "dancing",
    "idol",
}

STRONG_MUSICAL_TERMS = {
    "concert",
    "dance",
    "dancing",
    "idol",
    "jazz",
    "performance",
    "performer",
    "show",
    "singer",
    "singing",
    "song",
    "stage",
}

TRAGIC_TERMS = {
    "tragedy",
    "tragic",
    "tearjerker",
    "heartbreak",
    "heartbreaking",
    "grief",
    "loss",
    "death",
    "dying",
    "mourning",
    "depressing",
    "melancholy",
    "suicide",
}

COMFORT_TERMS = {
    "comfort",
    "comforting",
    "easy watch",
    "after work",
    "low stress",
    "heartwarming",
    "warm",
    "uplifting",
    "cozy",
}

DATE_NIGHT_TERMS = {
    "for couple",
    "date night",
    "couple",
}

FAMILY_FRIENDLY_TERMS = {
    "family watch",
    "family friendly",
    "friendly for family watch",
    "all ages",
}

CHILDISH_AVOIDANCE_TERMS = {
    "not childish",
    "not too childish",
    "not kiddy",
    "not too kiddy",
    "not for little kids",
}

HIGH_RATED_TERMS = {
    "high rated",
    "high-rated",
    "high imdb rating",
    "high imbd rating",
    "top rated",
    "top-rated",
    "critically acclaimed",
    "acclaimed",
}

THEME_TERMS = {
    "superhero": {"avenger", "avengers", "comic book", "dc", "marvel", "superhero", "superheroes"},
    "world_war_ii": {
        "allied",
        "battle of",
        "nazis",
        "nazi",
        "second world war",
        "world war 2",
        "world war ii",
        "ww2",
        "wwii",
    },
}

LANGUAGE_TERMS = {
    "ar": {"arabic"},
    "de": {"german"},
    "en": {"american", "british", "english"},
    "es": {"espanol", "spanish"},
    "fr": {"french"},
    "hi": {"hindi"},
    "it": {"italian"},
    "ja": {"anime", "japanese"},
    "ko": {"korean"},
    "pt": {"portuguese"},
    "ru": {"russian"},
    "zh": {"chinese", "mandarin"},
}

COUNTRY_TERMS = {
    "australia": {"australia", "australian"},
    "china": {"china", "chinese"},
    "france": {"france", "french"},
    "india": {"india", "indian", "bollywood"},
    "italy": {"italian", "italy"},
    "japan": {"japan", "japanese"},
    "mexico": {"mexico", "mexican"},
    "south korea": {"korea", "korean", "south korea"},
    "spain": {"spain", "spanish"},
    "united kingdom": {"britain", "british", "uk", "united kingdom"},
    "united states of america": {"america", "american", "hollywood", "united states", "usa"},
}

TYPO_REPLACEMENTS = {
    "imbd": "imdb",
    "realse": "release",
    "relase": "release",
    "super hero": "superhero",
    "super heros": "superheroes",
    "superheros": "superheroes",
    "supeheroes": "superheroes",
    "super heroe": "superhero",
}

LANGUAGE_TO_COUNTRIES = {
    "ar": {"arabia", "egypt", "saudi arabia", "united arab emirates"},
    "de": {"germany", "austria"},
    "en": {"united states of america", "united kingdom", "australia", "canada"},
    "es": {"spain", "mexico"},
    "fr": {"france", "belgium", "canada"},
    "hi": {"india"},
    "it": {"italy"},
    "ja": {"japan"},
    "ko": {"south korea"},
    "pt": {"portugal", "brazil"},
    "ru": {"russia"},
    "zh": {"china", "hong kong", "taiwan"},
}

NEGATION_WORDS = {"avoid", "exclude", "excluding", "no", "not", "skip", "without"}
SHORT_RUNTIME_TERMS = {"short", "under 2 hours", "under two hours", "quick"}
LONG_RUNTIME_TERMS = {"epic", "long", "sweeping"}
FAST_PACED_TERMS = {"adrenaline", "brisk", "fast paced", "fast-paced"}
RECENT_TERMS = {"current", "latest", "modern", "new", "recent"}
OLDER_TERMS = {"classic", "older", "old school", "retro", "throwback"}
CLASSIC_TERMS = {"classic", "older classic", "old school", "retro"}

RECOMMENDATION_CACHE: OrderedDict[
    tuple[str, tuple[str, ...], tuple[int, ...]], dict[str, object]
] = OrderedDict()
TMDB_DETAILS_CACHE: dict[int, dict] = {}


def _normalize_text(value) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value).casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    for typo, replacement in TYPO_REPLACEMENTS.items():
        text = re.sub(rf"\b{re.escape(typo)}\b", replacement, text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in text.split()
        if len(token) > 1 and token not in STOPWORDS
    ]


def _contains_phrase(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    return f" {phrase} " in f" {text} "


def _is_negated(text: str, phrase: str) -> bool:
    escaped = re.escape(phrase)
    negation_pattern = "|".join(re.escape(word) for word in sorted(NEGATION_WORDS))
    pattern = rf"(?:^| )(?:{negation_pattern})(?: [a-z0-9]+){{0,3}} {escaped}(?: |$)"
    return re.search(pattern, text) is not None


def _title_lookup_keys(value: str) -> set[str]:
    normalized = _normalize_text(value)
    if not normalized:
        return set()
    keys = {normalized}
    compact = normalized.replace(" ", "")
    if compact:
        keys.add(compact)
    for prefix in ("the ", "a ", "an "):
        if normalized.startswith(prefix):
            unprefixed = normalized[len(prefix):].strip()
            keys.add(unprefixed)
            keys.add(unprefixed.replace(" ", ""))
    return {key for key in keys if key}


def _split_csv_field(value) -> tuple[str, ...]:
    if value is None or pd.isna(value):
        return ()
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def _extract_categorical_signals(
    text: str, mapping: dict[str, set[str]]
) -> tuple[set[str], set[str]]:
    positive: set[str] = set()
    negative: set[str] = set()
    for label, phrases in mapping.items():
        for phrase in phrases:
            normalized_phrase = _normalize_text(phrase)
            if not _contains_phrase(text, normalized_phrase):
                continue
            if _is_negated(text, normalized_phrase):
                negative.add(label)
            else:
                positive.add(label)
    return positive, negative


def _extract_preference_signals(preferences: str) -> dict:
    normalized = _normalize_text(preferences)
    preferred_genres, avoided_genres = _extract_categorical_signals(
        normalized, GENRE_TERMS
    )
    preferred_moods, avoided_moods = _extract_categorical_signals(
        normalized, MOOD_TERMS
    )
    preferred_themes, avoided_themes = _extract_categorical_signals(
        normalized, THEME_TERMS
    )
    preferred_languages, avoided_languages = _extract_categorical_signals(
        normalized, LANGUAGE_TERMS
    )
    preferred_countries, avoided_countries = _extract_categorical_signals(
        normalized, COUNTRY_TERMS
    )
    return {
        "text": normalized,
        "tokens": _tokenize(normalized),
        "preferred_genres": preferred_genres,
        "avoided_genres": avoided_genres,
        "preferred_moods": preferred_moods,
        "avoided_moods": avoided_moods,
        "preferred_themes": preferred_themes,
        "avoided_themes": avoided_themes,
        "preferred_languages": preferred_languages,
        "avoided_languages": avoided_languages,
        "preferred_countries": preferred_countries,
        "avoided_countries": avoided_countries,
        "wants_musical": any(_contains_phrase(normalized, _normalize_text(term)) for term in MUSICAL_QUERY_TERMS),
        "wants_date_night": any(_contains_phrase(normalized, _normalize_text(term)) for term in DATE_NIGHT_TERMS),
        "avoids_sad": any(
            phrase in normalized
            for phrase in (
                "not too sad",
                "not sad",
                "not too tragic",
                "not tragic",
                "not heartbreaking",
            )
        ),
        "wants_comfort": any(_contains_phrase(normalized, _normalize_text(term)) for term in COMFORT_TERMS),
        "wants_family_friendly": any(
            _contains_phrase(normalized, _normalize_text(term)) for term in FAMILY_FRIENDLY_TERMS
        ),
        "avoids_childish": any(
            _contains_phrase(normalized, _normalize_text(term)) for term in CHILDISH_AVOIDANCE_TERMS
        ),
        "wants_high_rated": any(
            _contains_phrase(normalized, _normalize_text(term)) for term in HIGH_RATED_TERMS
        ),
        "wants_short": any(_contains_phrase(normalized, _normalize_text(term)) for term in SHORT_RUNTIME_TERMS),
        "wants_long": any(_contains_phrase(normalized, _normalize_text(term)) for term in LONG_RUNTIME_TERMS),
        "wants_fast": any(_contains_phrase(normalized, _normalize_text(term)) for term in FAST_PACED_TERMS),
        "prefers_recent": any(_contains_phrase(normalized, term) for term in RECENT_TERMS),
        "prefers_older": any(_contains_phrase(normalized, term) for term in OLDER_TERMS),
        "wants_classic": any(_contains_phrase(normalized, term) for term in CLASSIC_TERMS),
    }


def _quality_prior(vote_average: float, vote_count: float, popularity: float) -> float:
    return (
        0.35 * float(vote_average or 0)
        + 0.28 * math.log1p(max(float(vote_count or 0), 0))
        + 0.12 * math.log1p(max(float(popularity or 0), 0))
    )


def _build_movie_document(row) -> str:
    parts = [
        getattr(row, "title", ""),
        getattr(row, "genres", ""),
        getattr(row, "keywords", ""),
        getattr(row, "overview", ""),
        getattr(row, "tagline", ""),
        getattr(row, "director", ""),
        getattr(row, "top_cast", ""),
    ]
    return _normalize_text(" ".join(str(part) for part in parts if part and not pd.isna(part)))


def _tmdb_auth_headers() -> tuple[dict[str, str], dict[str, str]]:
    bearer_token = os.getenv("TMDB_BEARER_TOKEN", "").strip()
    api_key = os.getenv("TMDB_API_KEY", "").strip()
    headers = {"accept": "application/json"}
    query_params: dict[str, str] = {}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    elif api_key:
        query_params["api_key"] = api_key
    return headers, query_params


def _tmdb_get_json(path: str, query_params: dict[str, str] | None = None) -> dict | None:
    headers, auth_params = _tmdb_auth_headers()
    params = dict(auth_params)
    if query_params:
        params.update(query_params)
    if not headers.get("Authorization") and "api_key" not in params:
        return None

    url = f"https://api.themoviedb.org/3{path}"
    if params:
        url = f"{url}?{parse.urlencode(params)}"

    req = request.Request(url, headers=headers, method="GET")
    try:
        with request.urlopen(req, timeout=TMDB_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except (error.HTTPError, error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return None


def _fetch_tmdb_movie_details(tmdb_id: int) -> dict | None:
    if tmdb_id in TMDB_DETAILS_CACHE:
        return TMDB_DETAILS_CACHE[tmdb_id]

    payload = _tmdb_get_json(
        f"/movie/{tmdb_id}",
        {
            "append_to_response": "keywords,credits",
            "language": "en-US",
        },
    )
    if payload is None:
        return None

    genres = ", ".join(
        item["name"] for item in payload.get("genres", []) if item.get("name")
    )
    keyword_items = payload.get("keywords", {}).get("keywords", [])
    keywords = ", ".join(
        item["name"] for item in keyword_items if item.get("name")
    )
    credits = payload.get("credits", {})
    cast_names = ", ".join(
        person.get("name", "")
        for person in credits.get("cast", [])[:5]
        if person.get("name")
    )
    director = ""
    for crew_member in credits.get("crew", []):
        if crew_member.get("job") == "Director" and crew_member.get("name"):
            director = crew_member["name"]
            break

    enriched = {
        "title": payload.get("title") or "",
        "year": int(str(payload.get("release_date") or "0")[:4] or 0),
        "runtime_min": int(payload.get("runtime") or 0),
        "genres_text": genres,
        "overview": str(payload.get("overview") or "").strip(),
        "tagline": str(payload.get("tagline") or "").strip(),
        "director": director,
        "top_cast": cast_names,
        "keywords": keywords,
    }
    TMDB_DETAILS_CACHE[tmdb_id] = enriched
    return enriched


def _enrich_candidate_with_tmdb(candidate: dict) -> dict:
    tmdb_payload = _fetch_tmdb_movie_details(int(candidate["tmdb_id"]))
    if not tmdb_payload:
        return candidate

    enriched = dict(candidate)
    for key, value in tmdb_payload.items():
        if value:
            enriched[key] = value
    return enriched


def _prepare_candidates(frame: pd.DataFrame) -> list[dict]:
    candidates: list[dict] = []
    for row in frame.itertuples(index=False):
        genres = frozenset(_split_csv_field(getattr(row, "genres", "")))
        document = _build_movie_document(row)
        tokens = Counter(_tokenize(document))
        token_norm = math.sqrt(sum(count * count for count in tokens.values())) or 1.0
        candidate = {
            "tmdb_id": int(getattr(row, "tmdb_id")),
            "title": str(getattr(row, "title", "")).strip(),
            "title_norm": _normalize_text(getattr(row, "title", "")),
            "title_keys": frozenset(_title_lookup_keys(getattr(row, "title", ""))),
            "year": int(getattr(row, "year", 0) or 0),
            "runtime_min": int(getattr(row, "runtime_min", 0) or 0),
            "genres": genres,
            "genres_text": ", ".join(genres),
            "overview": str(getattr(row, "overview", "") or "").strip(),
            "tagline": str(getattr(row, "tagline", "") or "").strip(),
            "director": str(getattr(row, "director", "") or "").strip(),
            "top_cast": str(getattr(row, "top_cast", "") or "").strip(),
            "keywords": str(getattr(row, "keywords", "") or "").strip(),
            "original_language": str(getattr(row, "original_language", "") or "").strip().lower(),
            "countries": frozenset(
                _normalize_text(country)
                for country in _split_csv_field(getattr(row, "production_countries", ""))
                if _normalize_text(country)
            ),
            "countries_text": str(getattr(row, "production_countries", "") or "").strip(),
            "vote_average": float(getattr(row, "vote_average", 0) or 0),
            "vote_count": float(getattr(row, "vote_count", 0) or 0),
            "popularity": float(getattr(row, "popularity", 0) or 0),
            "document": document,
            "token_counts": tokens,
            "token_norm": token_norm,
            "is_superhero": any(
                _contains_phrase(document, _normalize_text(term))
                for term in THEME_TERMS["superhero"]
            ),
            "is_dark": any(
                _contains_phrase(document, _normalize_text(term))
                for term in MOOD_TERMS["dark"]
            ) or "Thriller" in genres or "Horror" in genres,
            "is_light": any(
                _contains_phrase(document, _normalize_text(term))
                for term in MOOD_TERMS["light"] | MOOD_TERMS["feel_good"] | MOOD_TERMS["funny"]
            ) or bool({"Comedy", "Family", "Animation"} & genres),
        }
        candidate["is_musical"] = (
            "Music" in genres
            or any(
                _contains_phrase(document, _normalize_text(term))
                for term in MUSICAL_CANDIDATE_TERMS
            )
        )
        candidate["musical_strength"] = 0.0
        if "Music" in genres:
            candidate["musical_strength"] += 2.5
        strong_matches = sum(
            1
            for term in STRONG_MUSICAL_TERMS
            if _contains_phrase(document, _normalize_text(term))
        )
        candidate["musical_strength"] += min(2.0, 0.8 * strong_matches)
        if _contains_phrase(document, " musical "):
            candidate["musical_strength"] += 0.4
        candidate["is_tragic"] = any(
            _contains_phrase(document, _normalize_text(term))
            for term in TRAGIC_TERMS
        )
        candidate["is_family_friendly"] = bool({"Family", "Animation"} & genres) or any(
            _contains_phrase(document, _normalize_text(term))
            for term in {"family", "all ages", "kid", "kids", "children"}
        )
        candidate["is_childish"] = any(
            _contains_phrase(document, _normalize_text(term))
            for term in {"preschool", "little kids", "toy", "talking animal", "childish"}
        )
        candidate["is_comfort_watch"] = (
            candidate["is_light"]
            and not candidate["is_dark"]
            and 0 < candidate["runtime_min"] <= 125
        )
        candidate["quality"] = _quality_prior(
            candidate["vote_average"], candidate["vote_count"], candidate["popularity"]
        )
        candidates.append(candidate)
    return candidates


CANDIDATES = _prepare_candidates(TOP_MOVIES)
CANDIDATE_BY_ID = {candidate["tmdb_id"]: candidate for candidate in CANDIDATES}
TITLE_INDEX: dict[str, list[dict]] = {}
DOCUMENT_FREQUENCY = Counter()
for candidate in CANDIDATES:
    for key in candidate["title_keys"]:
        TITLE_INDEX.setdefault(key, []).append(candidate)
    DOCUMENT_FREQUENCY.update(candidate["token_counts"].keys())

TOKEN_IDF = {
    token: math.log((1 + len(CANDIDATES)) / (1 + frequency)) + 1.0
    for token, frequency in DOCUMENT_FREQUENCY.items()
}

for candidate in CANDIDATES:
    tfidf_weights = {
        token: count * TOKEN_IDF[token]
        for token, count in candidate["token_counts"].items()
    }
    candidate["tfidf_weights"] = tfidf_weights
    candidate["tfidf_norm"] = math.sqrt(
        sum(weight * weight for weight in tfidf_weights.values())
    ) or 1.0


def _normalize_history_ids(history_ids) -> set[int]:
    normalized: set[int] = set()
    for value in history_ids or []:
        try:
            normalized.add(int(value))
        except (TypeError, ValueError):
            continue
    return normalized


def _normalize_history_titles(history) -> set[str]:
    normalized: set[str] = set()
    for title in history or []:
        normalized.update(_title_lookup_keys(title))
    return normalized


def _resolve_history_candidates(history: list[str], history_ids: list[int] | None) -> list[dict]:
    resolved: dict[int, dict] = {}
    for history_id in history_ids or []:
        try:
            candidate = CANDIDATE_BY_ID.get(int(history_id))
        except (TypeError, ValueError):
            candidate = None
        if candidate:
            resolved[candidate["tmdb_id"]] = candidate

    for title in history or []:
        for key in _title_lookup_keys(title):
            matches = TITLE_INDEX.get(key, [])
            if not matches:
                continue
            best_match = max(matches, key=lambda candidate: (candidate["quality"], candidate["year"]))
            resolved[best_match["tmdb_id"]] = best_match

    return list(resolved.values())


def _summarize_history(history_candidates: list[dict]) -> dict:
    genres = Counter()
    for candidate in history_candidates:
        genres.update(candidate["genres"])
    return {"count": len(history_candidates), "genres": genres}


def _build_query_tfidf(preferences: str) -> tuple[dict[str, float], float]:
    token_counts = Counter(_tokenize(_normalize_text(preferences)))
    tfidf_weights = {
        token: count * TOKEN_IDF.get(token, 1.0)
        for token, count in token_counts.items()
    }
    norm = math.sqrt(sum(weight * weight for weight in tfidf_weights.values())) or 1.0
    return tfidf_weights, norm


def _tfidf_similarity(query_weights: dict[str, float], query_norm: float, candidate: dict) -> float:
    if not query_weights:
        return 0.0
    dot = 0.0
    candidate_weights = candidate["tfidf_weights"]
    for token, query_weight in query_weights.items():
        dot += query_weight * candidate_weights.get(token, 0.0)
    return dot / (query_norm * candidate["tfidf_norm"])


def _genre_bonus(candidate: dict, signals: dict) -> float:
    score = 0.0
    score += 1.7 * len(candidate["genres"] & signals["preferred_genres"])
    score -= 2.5 * len(candidate["genres"] & signals["avoided_genres"])
    if signals["preferred_genres"] and not (candidate["genres"] & signals["preferred_genres"]):
        score -= 0.75
    return score


def _theme_bonus(candidate: dict, signals: dict) -> float:
    score = 0.0
    if "superhero" in signals["preferred_themes"] and candidate["is_superhero"]:
        score += 1.8
    if "superhero" in signals["avoided_themes"] and candidate["is_superhero"]:
        score -= 6.0
    if "world_war_ii" in signals["preferred_themes"]:
        if "War" in candidate["genres"] or "History" in candidate["genres"]:
            score += 3.2
        wwii_terms = THEME_TERMS["world_war_ii"]
        if any(
            _contains_phrase(candidate["document"], _normalize_text(term))
            for term in wwii_terms
        ):
            score += 2.0
        if candidate["is_superhero"] or "Science Fiction" in candidate["genres"]:
            score -= 2.5
    return score


def _mood_bonus(candidate: dict, signals: dict) -> float:
    score = 0.0
    for mood in signals["preferred_moods"]:
        if mood == "dark" and candidate["is_dark"]:
            score += 1.2
        elif mood == "light" and candidate["is_light"]:
            score += 1.2
        elif any(
            _contains_phrase(candidate["document"], _normalize_text(term))
            for term in MOOD_TERMS[mood]
        ):
            score += 1.0
    for mood in signals["avoided_moods"]:
        if mood == "dark" and candidate["is_dark"]:
            score -= 3.2
        elif mood == "light" and candidate["is_light"]:
            score -= 1.8
        elif any(
            _contains_phrase(candidate["document"], _normalize_text(term))
            for term in MOOD_TERMS[mood]
        ):
            score -= 1.6
    return score


def _runtime_bonus(candidate: dict, signals: dict) -> float:
    runtime = candidate["runtime_min"]
    score = 0.0
    if signals["wants_short"]:
        if 0 < runtime <= 105:
            score += 0.9
        elif runtime >= 145:
            score -= 0.8
    if signals["wants_long"]:
        if runtime >= 135:
            score += 0.8
        elif 0 < runtime <= 95:
            score -= 0.5
    if signals["wants_fast"]:
        if 0 < runtime <= 125:
            score += 0.6
    return score


def _locale_bonus(candidate: dict, signals: dict) -> float:
    score = 0.0
    preferred_language_countries = set().union(
        *(LANGUAGE_TO_COUNTRIES.get(code, set()) for code in signals["preferred_languages"])
    ) if signals["preferred_languages"] else set()

    if signals["preferred_languages"]:
        if candidate["original_language"] in signals["preferred_languages"]:
            score += 4.0
        elif candidate["countries"] & preferred_language_countries:
            score += 1.8
        else:
            score -= 2.0
    if candidate["original_language"] in signals["avoided_languages"]:
        score -= 3.0

    if signals["preferred_countries"]:
        if candidate["countries"] & signals["preferred_countries"]:
            score += 3.2
        else:
            score -= 1.6
    if candidate["countries"] & signals["avoided_countries"]:
        score -= 2.5
    if (
        signals["preferred_languages"]
        and signals["preferred_countries"]
        and candidate["original_language"] in signals["preferred_languages"]
        and candidate["countries"] & signals["preferred_countries"]
    ):
        score += 1.2
    return score


def _year_bonus(candidate: dict, signals: dict) -> float:
    year = candidate["year"]
    score = 0.0
    if signals["prefers_recent"]:
        if year >= 2020:
            score += 1.8 + (year - 2020) * 0.08
        elif year >= 2017:
            score += 0.5
        else:
            score -= min(4.5, (2020 - year) * 0.18)
    if signals["prefers_older"]:
        if year <= 2005:
            score += 1.6 + (2005 - year) * 0.03
        elif year <= 2012:
            score += 0.3
        else:
            score -= min(3.5, (year - 2005) * 0.12)
    if signals["wants_classic"]:
        if year <= 1999:
            score += 6.0
        elif year <= 2005:
            score += 2.4
        elif year <= 2010:
            score -= 0.6
        else:
            score -= min(9.0, (year - 2000) * 0.36)
    return score


def _history_bonus(candidate: dict, history_summary: dict) -> float:
    if not history_summary["count"]:
        return 0.0
    overlap = sum(
        min(history_summary["genres"][genre], 2)
        for genre in candidate["genres"]
    )
    return min(1.2, overlap * 0.35)


def _hard_constraint_penalty(candidate: dict, signals: dict) -> float:
    penalty = 0.0
    if "superhero" in signals["avoided_themes"] and candidate["is_superhero"]:
        penalty -= 8.0
    if signals["prefers_recent"] and candidate["year"] < 2015:
        penalty -= 2.0
    if signals["wants_classic"] and candidate["year"] > 2010:
        penalty -= 4.5
    if signals["wants_classic"] and candidate["is_superhero"]:
        penalty -= 3.0
    if signals["wants_classic"] and {"Action", "Science Fiction"} <= candidate["genres"]:
        penalty -= 1.5
    if "light" in signals["preferred_moods"] and candidate["is_dark"]:
        penalty -= 2.8
    if "world_war_ii" in signals["preferred_themes"]:
        if "War" not in candidate["genres"] and "History" not in candidate["genres"]:
            penalty -= 2.5
        if candidate["is_superhero"]:
            penalty -= 5.0
        if candidate["title_norm"].startswith("star wars") or "warcraft" in candidate["title_norm"]:
            penalty -= 4.0
    if signals["preferred_languages"] and candidate["original_language"] not in signals["preferred_languages"]:
        preferred_language_countries = set().union(
            *(LANGUAGE_TO_COUNTRIES.get(code, set()) for code in signals["preferred_languages"])
        )
        if not (candidate["countries"] & preferred_language_countries):
            penalty -= 2.8
    if signals["preferred_countries"] and not (candidate["countries"] & signals["preferred_countries"]):
        penalty -= 2.0
    return penalty


def _special_request_bonus(candidate: dict, signals: dict) -> float:
    score = 0.0

    if signals["wants_musical"]:
        if candidate["musical_strength"] >= 2.0:
            score += 3.4
        elif candidate["musical_strength"] >= 1.0:
            score += 1.4
        else:
            score -= 2.2

    if signals["wants_date_night"]:
        if "Romance" in candidate["genres"] and not candidate["is_tragic"]:
            score += 2.2
        elif "Romance" in candidate["genres"]:
            score += 0.6
        if candidate["is_light"]:
            score += 0.8
        if candidate["is_tragic"]:
            score -= 2.8

    if signals["avoids_sad"]:
        if candidate["is_tragic"]:
            score -= 3.4
        elif candidate["is_light"]:
            score += 0.7

    if signals["wants_comfort"]:
        if candidate["is_comfort_watch"]:
            score += 3.1
        if candidate["runtime_min"] and candidate["runtime_min"] > 135:
            score -= 2.4
        if candidate["is_dark"]:
            score -= 3.4
        if "Drama" in candidate["genres"] and not candidate["is_light"]:
            score -= 1.0

    if signals["wants_family_friendly"]:
        if candidate["is_family_friendly"]:
            score += 2.2
        else:
            score -= 2.0

    if signals["avoids_childish"]:
        if candidate["is_childish"]:
            score -= 1.8
        elif "Fantasy" in candidate["genres"] or "Adventure" in candidate["genres"]:
            score += 0.5

    if signals["wants_high_rated"]:
        if candidate["vote_average"] >= 8.0:
            score += 1.8
        elif candidate["vote_average"] >= 7.4:
            score += 1.0
        elif candidate["vote_average"] < 6.8:
            score -= 1.4

    if (
        "Thriller" in signals["preferred_genres"]
        and ("Horror" in signals["avoided_genres"] or "scary" in signals["avoided_moods"])
    ):
        if "Thriller" in candidate["genres"] and "Horror" not in candidate["genres"]:
            score += 1.6
        if "Horror" in candidate["genres"] or candidate["is_dark"]:
            score -= 1.6

    return score


def _score_candidate(
    candidate: dict,
    query_weights: dict[str, float],
    query_norm: float,
    signals: dict,
    history_summary: dict,
) -> float:
    similarity = _tfidf_similarity(query_weights, query_norm, candidate)
    return (
        8.0 * similarity
        + candidate["quality"]
        + _genre_bonus(candidate, signals)
        + _theme_bonus(candidate, signals)
        + _mood_bonus(candidate, signals)
        + _runtime_bonus(candidate, signals)
        + _locale_bonus(candidate, signals)
        + _year_bonus(candidate, signals)
        + _history_bonus(candidate, history_summary)
        + _special_request_bonus(candidate, signals)
        + _hard_constraint_penalty(candidate, signals)
    )


def _geo_preference_notice(signals: dict) -> str:
    labels: list[str] = []
    if signals["preferred_languages"]:
        labels.extend(sorted(signals["preferred_languages"]))
    if signals["preferred_countries"]:
        labels.extend(sorted(signals["preferred_countries"]))
    if not labels:
        return ""
    nice = ", ".join(labels)
    return f"I couldn't find a strong match for the requested language/country preference ({nice}) in this catalog, so this is the closest overall fit."


def _preferred_language_country_targets(signals: dict) -> set[str]:
    targets: set[str] = set()
    for code in signals["preferred_languages"]:
        targets.update(LANGUAGE_TO_COUNTRIES.get(code, set()))
    return targets


def _locale_pools(candidate_pool: list[dict], signals: dict) -> tuple[list[dict], list[dict], list[dict]]:
    language_country_targets = _preferred_language_country_targets(signals)

    language_pool = []
    if signals["preferred_languages"]:
        language_pool = [
            candidate
            for candidate in candidate_pool
            if (
                candidate["original_language"] in signals["preferred_languages"]
                or bool(candidate["countries"] & language_country_targets)
            )
        ]

    country_pool = []
    if signals["preferred_countries"]:
        country_pool = [
            candidate
            for candidate in candidate_pool
            if candidate["countries"] & signals["preferred_countries"]
        ]

    strict_pool = []
    if signals["preferred_languages"] and signals["preferred_countries"]:
        strict_pool = [
            candidate
            for candidate in candidate_pool
            if (
                candidate["original_language"] in signals["preferred_languages"]
                and bool(candidate["countries"] & signals["preferred_countries"])
            )
        ]

    return strict_pool, language_pool, country_pool


def _choose_movie(
    preferences: str, history: list[str], history_ids: list[int] | None = None
) -> tuple[dict, str]:
    signals = _extract_preference_signals(preferences)
    seen_ids = _normalize_history_ids(history_ids)
    seen_titles = _normalize_history_titles(history)
    history_candidates = _resolve_history_candidates(history, history_ids)
    history_summary = _summarize_history(history_candidates)
    query_weights, query_norm = _build_query_tfidf(preferences)

    if signals["wants_classic"]:
        candidate_pool = [
            candidate for candidate in CANDIDATES if candidate["year"] and candidate["year"] <= 2015
        ]
        if len(candidate_pool) < 25:
            candidate_pool = [
                candidate for candidate in CANDIDATES if candidate["year"] and candidate["year"] <= 2018
            ]
        if not candidate_pool:
            candidate_pool = CANDIDATES
    else:
        candidate_pool = CANDIDATES

    geo_notice = ""
    if signals["preferred_languages"] or signals["preferred_countries"]:
        strict_pool, language_pool, country_pool = _locale_pools(candidate_pool, signals)
        if strict_pool:
            candidate_pool = strict_pool
        elif language_pool:
            candidate_pool = language_pool
        elif country_pool:
            candidate_pool = country_pool
        else:
            geo_notice = _geo_preference_notice(signals)

    ranked: list[tuple[float, dict]] = []
    for candidate in candidate_pool:
        if candidate["tmdb_id"] in seen_ids:
            continue
        if candidate["title_norm"] in seen_titles or candidate["title_keys"] & seen_titles:
            continue
        score = _score_candidate(
            candidate, query_weights, query_norm, signals, history_summary
        )
        ranked.append((score, candidate))

    if not ranked:
        raise ValueError("No unseen candidates available.")

    ranked.sort(
        key=lambda item: (
            item[0],
            item[1]["quality"],
            item[1]["vote_average"],
            item[1]["year"],
        ),
        reverse=True,
    )
    return ranked[0][1], geo_notice


def _truncate_text(text: str, limit: int = MAX_DESCRIPTION_CHARS) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "")).strip().strip('"')
    if len(cleaned) <= limit:
        return cleaned
    shortened = cleaned[: limit - 3].rsplit(" ", 1)[0].rstrip(" ,;:-.")
    if not shortened:
        shortened = cleaned[: limit - 3].rstrip(" ,;:-.")
    return f"{shortened}..."


def _top_stars(candidate: dict, limit: int = 2) -> list[str]:
    cast = [
        name.strip()
        for name in str(candidate.get("top_cast", "") or "").split(",")
        if name.strip()
    ]
    return cast[:limit]


def _metadata_intro(candidate: dict) -> str:
    title = candidate["title"]
    year = candidate.get("year")
    runtime = int(candidate.get("runtime_min") or 0)
    pieces = []
    if year:
        pieces.append(str(year))
    if runtime > 0:
        pieces.append(f"{runtime} min")
    details = ", ".join(pieces)
    intro = f"{title}"
    if details:
        intro += f" ({details})"

    stars = _top_stars(candidate, limit=2)
    if len(stars) == 2:
        intro += f", starring {stars[0]} and {stars[1]}."
    elif len(stars) == 1:
        intro += f", starring {stars[0]}."
    else:
        intro += "."
    return intro


def _rating_note(candidate: dict) -> str:
    vote_average = float(candidate.get("vote_average") or 0.0)
    vote_count = int(float(candidate.get("vote_count") or 0.0))
    if vote_average >= HIGH_RATING_MIN and vote_count > MIN_RATING_VOTES:
        return f"It is highly rated at {vote_average:.1f}/10 with over {vote_count:,} TMDB votes."
    return ""


def _compose_blurb(candidate: dict, body: str) -> str:
    parts = [_metadata_intro(candidate)]
    rating_note = _rating_note(candidate)
    if rating_note:
        parts.append(rating_note)
    if body:
        parts.append(body.strip())
    return _truncate_text(" ".join(part for part in parts if part))


def _list_reasons(candidate: dict, preferences: str) -> list[str]:
    signals = _extract_preference_signals(preferences)
    reasons: list[str] = []
    for genre in sorted(candidate["genres"] & signals["preferred_genres"]):
        reasons.append(genre.lower())
    for mood in sorted(signals["preferred_moods"]):
        if any(
            _contains_phrase(candidate["document"], _normalize_text(term))
            for term in MOOD_TERMS[mood]
        ):
            reasons.append(mood.replace("_", " "))
    if candidate["director"]:
        reasons.append(f"a {candidate['director']} direction style")
    if not reasons and candidate["genres"]:
        reasons.extend(genre.lower() for genre in list(sorted(candidate["genres"]))[:2])
    deduped: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        key = _normalize_text(reason)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(reason)
    return deduped[:3]


def _fallback_description(candidate: dict, preferences: str, prefix_notice: str = "") -> str:
    reasons = _list_reasons(candidate, preferences)
    overview = candidate["overview"]
    title = candidate["title"]
    if len(reasons) >= 2:
        fit_sentence = f"It fits because it leans into {reasons[0]} and {reasons[1]}."
    elif reasons:
        fit_sentence = f"It fits because it leans into {reasons[0]}."
    else:
        fit_sentence = "It stands out as one of the strongest unseen matches in the catalog."

    story_sentence = ""
    if overview:
        story_sentence = overview[:180].rstrip(" .") + "."

    prefix = f"{prefix_notice} " if prefix_notice else ""
    body = f"{prefix}{title} is a strong pick for this request. {fit_sentence} {story_sentence}"
    return _compose_blurb(candidate, body)


def _build_description_prompt(candidate: dict, preferences: str, prefix_notice: str = "") -> str:
    reasons = ", ".join(_list_reasons(candidate, preferences)) or "its strongest qualities"
    metadata_intro = _metadata_intro(candidate)
    rating_note = _rating_note(candidate) or "No rating note needed."
    return f"""You are writing one short persuasive movie recommendation blurb.

User preferences:
{preferences.strip() or "No extra preferences provided."}

Chosen movie:
- Title: {candidate["title"]}
- Year: {candidate["year"]}
- Genres: {candidate["genres_text"] or "Unknown"}
- Director: {candidate["director"] or "Unknown"}
- Top cast: {candidate["top_cast"] or "Unknown"}
- Tagline: {candidate["tagline"] or "None"}
- Overview: {candidate["overview"] or "No overview available."}
- Main fit reasons: {reasons}
- Required intro to preserve exactly at the start of the final blurb: {metadata_intro}
- Include this rating note only if it reads naturally and is relevant: {rating_note}

Instructions:
- Write 1 to 2 short sentences after the required intro.
- Explain why this movie fits the user's request.
- If there is a catalog limitation note below, acknowledge it briefly in the first sentence.
- Be warm and natural.
- Do not mention TMDB IDs, scoring, ranking, or that you are an AI.
- Do not repeat the title, year, runtime, stars, or rating note unless needed for fluency.
- Keep the full final blurb under 500 characters.
- Return only the body text that should come after the required intro and optional rating note.

Catalog limitation note:
{prefix_notice or 'none'}
"""


def _generate_description(candidate: dict, preferences: str, prefix_notice: str = "") -> str:
    candidate = _enrich_candidate_with_tmdb(candidate)
    api_key = os.getenv("OLLAMA_API_KEY", "").strip()
    if not api_key:
        return _fallback_description(candidate, preferences, prefix_notice)

    try:
        client = ollama.Client(
            host="https://ollama.com",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response = client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": _build_description_prompt(candidate, preferences, prefix_notice)}],
        )
        content = str(response.message.content).strip()
        if not content:
            return _fallback_description(candidate, preferences, prefix_notice)
        return _compose_blurb(candidate, content)
    except Exception:
        return _fallback_description(candidate, preferences, prefix_notice)


def _cache_key(
    preferences: str, history: list[str], history_ids: list[int] | None
) -> tuple[str, tuple[str, ...], tuple[int, ...]]:
    normalized_preferences = _normalize_text(preferences)
    normalized_titles = tuple(sorted(_normalize_history_titles(history)))
    normalized_ids = tuple(sorted(_normalize_history_ids(history_ids)))
    return normalized_preferences, normalized_titles, normalized_ids


def _cache_get(
    cache_key: tuple[str, tuple[str, ...], tuple[int, ...]]
) -> dict[str, object] | None:
    cached = RECOMMENDATION_CACHE.get(cache_key)
    if cached is None:
        return None
    RECOMMENDATION_CACHE.move_to_end(cache_key)
    return {
        "tmdb_id": int(cached["tmdb_id"]),
        "description": str(cached["description"]),
    }


def _cache_set(
    cache_key: tuple[str, tuple[str, ...], tuple[int, ...]], result: dict[str, object]
) -> None:
    RECOMMENDATION_CACHE[cache_key] = {
        "tmdb_id": int(result["tmdb_id"]),
        "description": str(result["description"]),
    }
    RECOMMENDATION_CACHE.move_to_end(cache_key)
    if len(RECOMMENDATION_CACHE) > CACHE_MAX_SIZE:
        RECOMMENDATION_CACHE.popitem(last=False)


def get_recommendation(
    preferences: str, history: list[str], history_ids: list[int] | None = None
) -> dict:
    """Return a dict with keys 'tmdb_id' (int) and 'description' (str)."""
    recommendation_cache_key = _cache_key(preferences, history, history_ids)
    cached = _cache_get(recommendation_cache_key)
    if cached is not None:
        return cached

    candidate, geo_notice = _choose_movie(preferences, history, history_ids)
    result = {
        "tmdb_id": int(candidate["tmdb_id"]),
        "description": _generate_description(candidate, preferences, geo_notice),
    }
    result["description"] = _truncate_text(str(result["description"]))
    _cache_set(recommendation_cache_key, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a local movie recommendation test."
    )
    parser.add_argument(
        "--preferences",
        type=str,
        help="User preferences text. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--history",
        type=str,
        help='Comma-separated watch history titles. Example: "The Avengers, Up"',
    )
    args = parser.parse_args()

    print("Movie recommender – type your preferences and press Enter.")
    print(
        "For watch history, enter comma-separated movie titles (or leave blank)."
    )

    preferences = (
        args.preferences.strip()
        if args.preferences and args.preferences.strip()
        else input("Preferences: ").strip()
    )
    history_raw = (
        args.history.strip()
        if args.history and args.history.strip()
        else input("Watch history (optional): ").strip()
    )
    history = (
        [t.strip() for t in history_raw.split(",") if t.strip()]
        if history_raw
        else []
    )

    print("\nThinking...\n")
    start = time.perf_counter()
    result = get_recommendation(preferences, history)
    print(result)
    elapsed = time.perf_counter() - start

    print(f"\nServed in {elapsed:.2f}s")
