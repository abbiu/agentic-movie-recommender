# Project Summary: Agentic Movie Recommender

## What We Built

We upgraded the starter movie recommender from a simple LLM prompt into a hybrid recommendation system:

- Python selects the movie using deterministic ranking.
- TF-IDF similarity helps match user preferences to movie metadata.
- Rule-based logic handles genres, moods, exclusions, recency, runtime, and watch history.
- Ollama writes the final recommendation blurb.
- TMDB API enriches the selected movie with fresher metadata before the blurb is written.
- Evaluation results are saved to `evaluation_results.csv`.

The FastAPI contract in `main.py` was left unchanged so the project remains compatible with the grader.

## Key Files

- `llm.py`: Main recommendation pipeline.
- `main.py`: FastAPI app and grading contract. We did not change this.
- `evaluate.py`: Evaluation script with 10 prompt scenarios and scoring.
- `evaluation_results.csv`: Latest evaluation output.
- `tmdb_top1000_movies.csv`: Local candidate pool. Returned `tmdb_id`s must come from this file.
- `.env.example`: Placeholder environment variables for API keys.
- `test.py`: Provided grading/test script.

## Recommendation Pipeline

The current system works in these steps:

1. Load all movies from `tmdb_top1000_movies.csv`.
2. Build a text document for each movie from title, genres, keywords, overview, tagline, director, and cast.
3. Compute TF-IDF weights for the movie corpus at import time.
4. Parse the user's preference text for:
   - preferred genres
   - avoided genres
   - mood and tone words
   - runtime hints
   - recent or older movie hints
   - superhero preference or exclusion
5. Normalize watch history and exclude watched movies by:
   - `tmdb_id`
   - title
   - punctuation/spacing-tolerant compact title keys
6. Score remaining movies using:
   - TF-IDF similarity
   - genre match
   - mood/tone match
   - quality prior from ratings/popularity
   - recency or older-movie preference
   - watch-history affinity
   - hard penalties for avoided traits
7. Select the best valid movie from the local CSV.
8. Optionally fetch live TMDB metadata for that selected movie.
9. Ask Ollama to write a short recommendation blurb.
10. Fall back to a deterministic description if the live API call fails.

## TMDB API Usage

TMDB is used as an enrichment layer, not as the primary candidate source.

This is important because `main.py` validates that every returned `tmdb_id` exists in `tmdb_top1000_movies.csv`. If TMDB recommends a movie outside the CSV, the API would reject it.

TMDB currently improves the final response by fetching:

- overview
- tagline
- runtime
- genres
- director
- top cast
- keywords

Recommended environment variables:

```bash
export OLLAMA_API_KEY="your_ollama_key"
export TMDB_API_KEY="your_tmdb_key"
```

You can also use:

```bash
export TMDB_BEARER_TOKEN="your_tmdb_bearer_token"
```

## Evaluation

We added `evaluate.py`, which runs 10 test prompts and writes `evaluation_results.csv`.

The evaluator checks:

- whether hard constraints were respected
- whether the result matches the user's intent
- whether watch history was avoided
- whether TMDB enrichment was available
- whether the generated description visibly used TMDB metadata

Current scoring columns:

- `constraint_score`
- `intent_score`
- `history_score`
- `tmdb_metadata_score`
- `description_score`
- `overall_score`

Run evaluation:

```bash
TMDB_API_KEY="your_tmdb_key" OLLAMA_API_KEY="your_ollama_key" python evaluate.py
```

Run the grader:

```bash
OLLAMA_API_KEY="your_ollama_key" python test.py
```

## Current Evaluation Takeaways

The recommender performs well on most tested scenarios:

- funny / feel-good movie
- dark suspense without horror
- mature romance
- recent serious sci-fi
- short comfort watch
- non-superhero action
- animated and funny
- scary movie
- light and recent

The main limitation is the "older classic" prompt. The CSV only contains movies from 2011 to 2026, so true classic movies are not available under the grader-safe candidate pool.

For that reason, the better evaluation wording is:

```text
Recommend one of the older movies available in this dataset.
```

instead of:

```text
Recommend an older classic movie.
```

## Watch History Improvements

We fixed a watch-history matching issue where:

```text
spider man into the spiderverse
```

did not match:

```text
Spider-Man: Into the Spider-Verse
```

The matcher now compares both normalized titles and compact no-space title keys, so punctuation and spacing differences are handled better.

Exact watched movies are excluded. Sequels or movies from the same franchise may still be recommended unless a future same-franchise avoidance feature is added.

## Known Limitations

- The recommender cannot return movies outside `tmdb_top1000_movies.csv` without changing `main.py`.
- True classic movie recommendations are limited by the dataset's year range.
- TMDB improves descriptions but does not expand the candidate pool.
- Same-franchise avoidance is not implemented yet.
- API keys should never be committed to GitHub.

## Suggested Next Improvements

- Add same-franchise avoidance using title prefixes or TMDB collection data.
- Adjust the evaluation prompt for older movies to match the dataset.
- Add a `same_franchise_score` to `evaluate.py`.
- Add more edge-case prompts, such as:
  - "I want animation but not Spider-Man."
  - "I want a funny movie, but not for kids."
  - "I want sci-fi, but not action-heavy."
  - "I want a family movie that is not animated."

