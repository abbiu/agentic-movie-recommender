# Project Summary: Agentic Movie Recommender

## What We Built

We turned the starter recommender into a hybrid recommendation system:

- Python chooses the movie with deterministic ranking.
- TF-IDF similarity matches user preferences and watch history to movie metadata.
- Rule-based logic handles genre, tone, exclusions, runtime, recency, history, language, and country.
- Ollama is used in two bounded roles:
  - semantic intent extraction before ranking
  - final recommendation blurb after a movie is selected
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
4. Parse the user prompt with deterministic rules for:
   - preferred and avoided genres
   - mood and tone cues
   - theme cues such as superhero and WWII
   - runtime and recency hints
   - language preferences
   - country preferences
   - musical/date-night/comfort/high-rated/family-friendly signals
5. Ask Ollama for a structured semantic interpretation of the prompt, including:
   - expanded retrieval query
   - soft semantic genres, moods, themes, languages, and countries
   - `must_terms`
   - `avoid_terms`
   - numeric constraints like `min_year`, `max_runtime`, and `min_vote_average`
   - whether history should matter more for a vague request
6. Merge the semantic interpretation conservatively:
   - keep explicit rule-based parsing as the source of truth
   - use semantic extraction mainly for:
     - richer retrieval text
     - numeric constraints
     - soft topical hints
     - vague-prompt history bias
7. Normalize watch history and exclude watched titles by:
   - `tmdb_id`
   - normalized titles
   - compact no-space title keys for typo and punctuation tolerance
8. Build a history profile from resolved watched titles:
   - genre summary
   - TF-IDF centroid of watched movie metadata
   - similarity to the closest watched item
9. Score unseen candidates with:
   - TF-IDF similarity to the interpreted prompt
   - TF-IDF similarity to watch history
   - genre bonus
   - mood bonus
   - theme bonus
   - runtime bonus
   - locale bonus
   - year bonus
   - quality prior from `vote_average`, `vote_count`, and `popularity`
   - special request bonuses for musical, comfort, date-night, high-rated, family-friendly, and low-scare thriller prompts
   - small semantic hint bonuses from Ollama’s extracted intent
   - hard penalties for violations such as superhero leakage, wrong locale, horror in low-scare prompts, or too-dark comfort picks
10. Optionally narrow the pool for strong locale requests using both:
   - `original_language`
   - `production_countries`
11. Select the best valid movie from the local CSV.
12. Optionally enrich that chosen movie with TMDB details.
13. Generate the final blurb with Ollama, or fall back to a deterministic template if Ollama is unavailable.

## Semantic Intent Extraction

One of the biggest recent improvements was moving from simple query expansion to a fuller semantic intent step.

The current semantic layer lets Ollama interpret the prompt first, rather than forcing the Python ranker to depend entirely on keyword overlap. The model now tries to identify:

- the main meaning of the prompt
- soft semantic tags
- topical `must_terms`
- explicit negatives
- numeric constraints like year/runtime/rating
- whether the prompt is vague enough that history should influence ranking more strongly

This made the architecture more scalable than endlessly hard-coding special cases. At the same time, we found that giving the model too much direct control over ranking made some results worse, so we tuned it back.

Current design decision:

- keep semantic extraction
- reduce its authority in ranking
- let it act as a hint layer, not the main decision-maker

That means the stable parts of the recommender still come from deterministic Python logic, while Ollama contributes higher-level prompt understanding.

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

We also upgraded history from a simple exclusion rule into a real similarity signal:

- watched titles are still excluded
- but if history is available, the ranker now also prefers movies that are similar to the user’s watched items
- this is especially strong for vague prompts like `recommend me something`

This is done through:

- genre overlap with watched titles
- TF-IDF similarity to a watched-history centroid
- TF-IDF similarity to the closest single watched movie

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

- locale requests are matched using both spoken language and production-country metadata when possible

## Evaluation

`evaluate.py` has been expanded from the earlier small suite into a broader manual benchmark.

It now tests 26 scenarios, including:

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
- typo-robust superhero prompts

Current evaluation scoring includes:

- `constraint_score`
- `intent_score`
- `history_score`
- `tmdb_metadata_score`
- `description_score`
- `description_quality_score`
- `blurb_format_score`
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

Latest local evaluation status:

- `23/26` results are `Reasonable fit`
- `3/26` are flagged by the evaluator
- current averages:
  - `constraint_score = 0.962`
  - `intent_score = 1.615`
  - `history_score = 1.000`
  - `description_score = 2.000`
  - `description_quality_score = 2.692`
  - `blurb_format_score = 5.000`
  - `overall_score = 13.269`

The current system is strongest on:

- watch-history exclusion
- history-aware similarity for vague prompts
- superhero exclusion in prompts like `not superheroes`
- typo robustness for several misspellings
- stable blurb formatting
- keeping the grader tests green while experimenting with smarter prompt interpretation

We also learned an important design lesson:

- semantic intent extraction is useful
- but if it has too much authority, it can destabilize ranking
- the best version so far is the reduced-authority setup where semantic parsing informs the ranker without overriding the more stable heuristics

## Known Limitations

- The recommender still cannot return movies outside `tmdb_top1000_movies.csv` without breaking the current grader contract.
- The `older classic` case is still constrained by the dataset itself, because the CSV only contains movies from 2011 onward.
- Same-franchise avoidance is still not implemented.
- The evaluator still flags `thriller but not too scary` even when the returned movie is arguably sensible, because the heuristic is intentionally simple.
- Buddy/team-dynamic understanding is still weaker than it should be for prompts like `superheroes and feel-good buddy cop stories`.
- Some semantically tricky requests still need better handling through softer structured hints rather than more hard-coded rules.
- TMDB enrichment depends on the environment having both the key and network access available at runtime.

## Suggested Next Improvements

- Strengthen social-dynamic matching for prompts about teams, duos, partners, buddy-cop energy, or ensemble casts.
- Add same-franchise avoidance using title-series logic or TMDB collection data.
- Refine the evaluator’s low-scare thriller heuristic.
- Add a dedicated score for locale correctness.
- Run evaluation with TMDB enabled and network access available when producing final report metrics.
- Continue using semantic intent extraction, but keep it as a low-authority hint layer unless a future version proves it can improve evaluation consistently.
