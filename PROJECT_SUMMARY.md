# Project Summary: Agentic Movie Recommender

## What We Built

We turned the starter recommender into a hybrid recommendation system:

- Python chooses the movie with deterministic ranking.
- TF-IDF similarity matches user preferences to movie metadata.
- Rule-based logic handles genre, tone, exclusions, runtime, recency, watch history, language, and country.
- Ollama is used only to write the final recommendation blurb.
- TMDB is used as an enrichment layer for the chosen movie, not as the primary candidate source.
- Evaluation results are written to `evaluation_results.csv`.

The FastAPI contract in `main.py` was left unchanged so the app stays compatible with the grader.

## Key Files

- `llm.py`: Main recommendation pipeline.
- `main.py`: FastAPI app and grader-safe API contract.
- `evaluate.py`: Manual evaluation suite and scoring.
- `evaluation_results.csv`: Latest evaluation output.
- `tmdb_top1000_movies.csv`: Local candidate pool.
- `test.py`: Provided grading script.
- `.env.example`: Placeholder environment variables.

## Recommendation Pipeline

The current flow is:

1. Load the local `tmdb_top1000_movies.csv` candidate pool.
2. Build a text document per movie using title, genres, keywords, overview, tagline, director, cast, language, and country metadata.
3. Compute TF-IDF features locally at import time.
4. Parse the user prompt for:
   - preferred and avoided genres
   - mood and tone cues
   - theme cues such as superhero and WWII
   - runtime and recency hints
   - language preferences
   - country preferences
   - musical/date-night/comfort/high-rated/family-friendly signals
5. Normalize watch history and exclude watched titles by:
   - `tmdb_id`
   - normalized titles
   - compact no-space title keys for typo and punctuation tolerance
6. Score unseen candidates with:
   - TF-IDF similarity
   - genre bonus
   - mood bonus
   - theme bonus
   - runtime bonus
   - locale bonus
   - year bonus
   - quality prior from `vote_average`, `vote_count`, and `popularity`
   - special request bonuses for musical, comfort, date-night, high-rated, family-friendly, and low-scare thriller prompts
   - hard penalties for violations such as superhero leakage, wrong locale, horror in low-scare prompts, or too-dark comfort picks
7. Optionally narrow the pool for strong locale requests using both:
   - `original_language`
   - `production_countries`
8. Select the best valid movie from the local CSV.
9. Optionally enrich that chosen movie with TMDB details.
10. Generate the final blurb with Ollama, or fall back to a deterministic template if Ollama is unavailable.

## TMDB API Usage

TMDB is used only after selection as an enrichment layer.

This matters because `main.py` validates that the returned `tmdb_id` must already exist in `tmdb_top1000_movies.csv`. That means TMDB cannot safely expand the candidate pool without changing the grader contract.

TMDB currently helps provide:

- overview
- tagline
- runtime
- genres
- director
- top cast
- keywords

Supported environment variables:

```bash
export OLLAMA_API_KEY="your_ollama_key"
export TMDB_API_KEY="your_tmdb_key"
```

Or:

```bash
export TMDB_BEARER_TOKEN="your_tmdb_bearer_token"
```

## Recommendation Blurb Format

The final blurb now has fixed metadata formatting before the explanation:

- movie title
- year
- duration
- up to 2 top stars

If the movie is strongly rated and has enough support, the blurb also includes a rating note:

- `vote_average >= 7.5`
- `vote_count > 200`

Example shape:

```text
La La Land (2016, 129 min), starring Ryan Gosling and Emma Stone. It is highly rated at 7.9/10 with over 17,968 TMDB votes. ...
```

This formatting is enforced in Python so it does not depend on the model remembering instructions perfectly.

## Watch History Improvements

We fixed a history-matching issue where user input like:

```text
spider man into the spiderverse
```

did not match:

```text
Spider-Man: Into the Spider-Verse
```

The matcher now compares:

- normalized titles
- compact no-space title keys

This makes history filtering much more robust for punctuation, spacing, and light formatting differences.

## Language and Country Handling

Locale preference handling is now stronger than earlier versions.

The recommender now:

- parses language requests such as `korean`, `japanese`, `french`
- parses country requests such as `South Korea`, `Japan`, `France`, `United States`
- uses `original_language`
- also uses `production_countries`
- narrows the candidate pool when there is a strong locale request
- falls back gracefully if the local catalog cannot satisfy that request

Example:

- `funny romantic movie in Korean` now returns a Korean-language, South Korea-produced match instead of a generic English romance

## Evaluation

`evaluate.py` has been expanded from the earlier small suite into a broader manual benchmark.

It now tests 25 scenarios, including:

- funny / feel-good
- dark suspense without horror
- mature romance
- recent serious sci-fi
- short comfort watch
- non-superhero action
- animated comedy
- older/classic request
- scary movie
- light and recent
- WWII historical action
- recent animation
- high-rated short sci-fi
- date-night romance that is not too sad
- musical romance
- family fantasy but not childish
- thriller but not too scary
- comfort movie after work
- family-friendly action adventure
- superhero + buddy dynamic
- typo robustness
- default English behavior
- explicit Korean-language behavior
- watch-history strictness
- description quality

Current evaluation scoring includes:

- `constraint_score`
- `intent_score`
- `history_score`
- `tmdb_metadata_score`
- `description_score`
- `description_quality_score`
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

The recommender now performs well on most tested scenarios, especially:

- superhero exclusion
- horror vs non-horror distinction
- watch history avoidance
- comfort-watch handling
- Korean language and country prompts
- musical/date-night prompts
- WWII prompts
- typo robustness for some noisy inputs

Current examples from the updated evaluation:

- `romantic movie for couple, not too sad` -> `La La Land`
- `love story, fun, musical, for couple` -> `The Idea of You`
- `comfort movie after work` -> `Inside Out`
- `funny romantic movie in Korean` -> `You Are the Apple of My Eye`

## Known Limitations

- The recommender still cannot return movies outside `tmdb_top1000_movies.csv` without breaking the current grader contract.
- The `older classic` case is still constrained by the dataset itself, because the CSV only contains movies from 2011 onward.
- Same-franchise avoidance is still not implemented.
- Some typo cases still need explicit normalization, for example misspellings like `superheros`.
- The evaluator contains a few heuristic checks that are intentionally simple and may still over-warn on sensible results.

## Suggested Next Improvements

- Add typo normalization for more malformed superhero and genre words.
- Add same-franchise avoidance using title-series logic or TMDB collection data.
- Refine the evaluator’s low-scare thriller heuristic.
- Add a dedicated score for locale correctness.
- Add more stress tests around:
  - foreign-language romance
  - sequel avoidance
  - franchise avoidance
  - “not for kids” vs family-friendly prompts
