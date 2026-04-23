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
]


OUTPUT_CSV = os.path.join(
    os.path.dirname(__file__), "evaluation_results.csv"
)


def quick_eval(case: dict, movie: dict) -> str:
    preferences = case["preferences"].lower()
    genres = movie["genres_text"].lower()
    metadata_text = (
        f"{movie['overview']} {movie['keywords']} {movie['tagline']}"
    ).lower()
    year = int(movie["year"] or 0)
    runtime = int(movie["runtime_min"] or 0)

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
    return "Reasonable fit"


def score_case(case: dict, movie: dict) -> tuple[int, int, int, str]:
    preferences = case["preferences"].lower()
    genres = movie["genres_text"].lower()
    metadata_text = (
        f"{movie['overview']} {movie['keywords']} {movie['tagline']}"
    ).lower()
    title_norm = movie["title"].strip().casefold()
    history_norm = {title.strip().casefold() for title in case["history"]}
    history_id_set = {int(value) for value in case["history_ids"] or []}
    year = int(movie["year"] or 0)
    runtime = int(movie["runtime_min"] or 0)

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
        overall_score = (
            constraint_score
            + intent_score
            + history_score
            + metadata_score
            + description_score
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
                "overall_score": overall_score,
                "quick_eval": assessment,
                "notes": notes,
                "tmdb_notes": metadata_notes,
                "description_notes": description_notes,
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
                "overall_score",
                "quick_eval",
                "notes",
                "tmdb_notes",
                "description_notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
