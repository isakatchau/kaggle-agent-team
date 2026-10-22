# Kaggle Agent Team

A team of five LLM agents that enters a Kaggle tabular competition on its own: it downloads the data, works out the task and metric, designs features, trains and ensembles models, decides when a result is worth one of the limited daily submissions, reads the leaderboard and plans the next round, until it reaches the target percentile (top 20% by default) or runs out of budget.

```bash
python -m kagent run --config configs/competitions/titanic.yaml
```

Switching competitions is a one-line YAML file: only the Kaggle slug is required.

## Overview

| Agent | Responsibility | Output (validated with pydantic) | Hard guardrails applied after the LLM answers |
|---|---|---|---|
| **Planner** | Reads the data profile, sample submission and the competition's evaluation page. Decides the task, metric, target, id column and submission format. | `CompetitionPlan` | Target must be a train-only column; id must be in the sample submission; metric must match the task; YAML overrides win |
| **Data** | Designs the feature plan: regex extraction from text, dates, frequency encoding, group sizes, fold-safe group target encoding, ratios, log transforms, log target. It picks transformations; a tool applies them. | `FeaturePlan` | Unknown columns, invalid regexes, bad pairs and group keys that are unique per row are removed; RMSLE forces a log target |
| **Modeling** | Proposes a batch of cross-validated experiments (model family and hyperparameters) and whether to ensemble. | `ExperimentBatch` | Params are clipped to bounds; duplicates of past runs are skipped; batch size is capped |
| **Evaluator** | Chooses the most trustworthy candidate and decides whether it deserves a submission. | `SubmitDecision` | No submission after the budget is spent, or unless CV beats the best submitted CV by `min_improvement` |
| **Leaderboard reviewer** (bonus) | Compares CV, public LB and percentile with the target. Diagnoses what is limiting the score and routes the next round to features, models or tuning, with concrete suggestions. | `Review` | Stops when the target is reached or iterations run out |

Every agent has a **deterministic heuristic** with the same output schema. It is used when no LLM is configured (`--llm mock`), when the LLM is unreachable, or when the LLM returns invalid JSON twice. The run never dies because of the model. The log records which path produced each decision (`source: openai:gpt-4o-mini`, `heuristic` or `heuristic-fallback`).

## Architecture

```mermaid
flowchart LR
    S((start)) --> P[Planner]
    P --> D[Data agent]
    D --> M[Modeling agent]
    M --> E[Evaluator]
    E --> R[Leaderboard reviewer]
    R -- "next_focus = features" --> D
    R -- "models / tuning" --> M
    R -- "stop" --> F[Finalize: summary, plots, index] --> X((end))

    subgraph tools [Tools: the only code that touches data or Kaggle]
      K[Kaggle CLI / offline platform]
      FB[Feature builder]
      T[CV trainer + greedy ensemble]
      W[Submission writer + pre-flight check]
    end
    P -.-> K
    D -.-> FB
    M -.-> T
    E -.-> W
    E -.-> K
```

- **Orchestration**: a LangGraph `StateGraph` (`src/kagent/orchestrator/graph.py`). Graph state holds only control flow (iteration, focus, done). DataFrames, features and models live in a shared `Workspace`.
- **LLM interface**: `LLMClient.complete_json(system, user) -> dict` (`src/kagent/llm/`). The OpenAI-compatible client uses JSON mode and talks to any compatible endpoint through `OPENAI_BASE_URL` (Groq's free tier by default in `.env.example`). Before a run, one tiny preflight call checks the key, URL and model, so a misconfiguration stops the run instead of silently degrading every agent to its heuristic. `summary.json` reports `llm_fallbacks`.
- **Agents never execute code they write.** They choose from typed menus (feature operations, model families, bounded params). This keeps the system safe and reproducible, and lets the heuristics and the LLM share one interface. The cost is expressiveness; see Limitations.
- **Observability**:
  - `runs/<competition>/<run_id>/events.jsonl`: every reasoning, decision, tool call, tool result, error and fallback.
  - `llm_calls.jsonl`: full prompts and responses, with tokens and latency.
  - `plan.json`, `feature_plan_v*.json`, `review_it*.json`, `experiments.csv`, `submissions.json` and `summary.json`.
  - `models/*.joblib`: fold models as checkpoints.
  - `plots/*.png`.
  - `runs/index.jsonl`: one line per attempt.

More detail is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```
src/kagent/
  agents/          planner, data, modeling, evaluator, reviewer, shared base, schemas
  llm/             interface, OpenAI client, scripted client for tests
  tools/           kaggle (CLI + offline), profile, features, train, metrics, submission, synthetic
  orchestrator/    LangGraph graph + Workspace
  observability/   JSONL run logger
  reporting/       matplotlib evidence
  cli.py           run | demo | list-playground
dashboard/app.py   Streamlit dashboard of attempts, scores and agent decisions
configs/competitions/*.yaml
tests/             offline unit + end-to-end tests
```

## How to run

### 1. Install (Python 3.11+)

```bash
python -m venv .venv && source .venv/bin/activate
make setup                     # pip install -r requirements.txt && pip install -e .
make test                      # offline, no keys needed
```

### 2. Credentials (kept outside the repo)

- **LLM:** `cp .env.example .env`, then set `OPENAI_API_KEY`. The example targets Groq (free key at console.groq.com, model `openai/gpt-oss-120b`). For OpenAI, remove `OPENAI_BASE_URL` and set `OPENAI_MODEL=gpt-4o-mini`.
- **Kaggle** (CLI 2.x): run `kaggle auth login` (OAuth in the browser, no token to manage), or generate a token at kaggle.com/settings/api and save it as `~/.kaggle/access_token` or export `KAGGLE_API_TOKEN`. Check with `python -m kagent kaggle competitions list`. The `kagent kaggle` wrapper is the official CLI verifying TLS against the OS certificate store (as all of `kagent` does through `truststore`), so it also works behind TLS-inspecting antivirus or corporate proxies; use `python -m kagent kaggle auth login` if plain `kaggle auth login` fails with `CERTIFICATE_VERIFY_FAILED`.
- **Competition rules:** open each competition page once and click *Join / I understand and accept*. The API returns 403 until you do.

### 3. Run

```bash
# offline end-to-end, no keys at all (synthetic competition with hidden labels)
python -m kagent demo

# real competition, LLM agents
python -m kagent run --config configs/competitions/titanic.yaml
python -m kagent run --config configs/competitions/house-prices.yaml

# without spending submissions (writes the CSVs only)
python -m kagent run --config configs/competitions/titanic.yaml --no-submit

# heuristic agents only (no OpenAI)
python -m kagent run --config configs/competitions/titanic.yaml --llm mock

make dashboard                 # http://localhost:8501
```

### 4. New competition

Copy `configs/competitions/playground.yaml` and set `competition:` to the slug in the URL (`kaggle.com/competitions/<slug>`). Run `python -m kagent list-playground` to find current episodes. Overrides (`metric`, `target`, `id_column`, `sample_rows`) are optional.

### 5. Docker

```bash
make docker-build
make docker-demo                         # offline
make docker-run COMP=titanic             # mounts ~/.kaggle read-only and passes .env
```

## Demo guide

1. **`make test`** (about 10 s) shows the tools, the guardrails and both offline loops working.
2. **`python -m kagent demo --task regression --target-percentile 0.05 --max-iterations 4`** shows the loop in action. The target is deliberately hard, so the reviewer keeps routing rounds: tuning → models → features. The evaluator holds back submissions that do not clear `min_improvement`, and the run stops on budget.
3. **`python -m kagent run --config configs/competitions/titanic.yaml`** runs on real Kaggle. The console shows each agent's reasoning and decision. Things to watch:
   - The Planner infers the metric from the evaluation page (or the sample submission when the page is unavailable).
   - The Data agent's regex and group-size choices for `Name`, `Ticket` and `Cabin`; this is where the LLM differs most from the heuristic.
   - The Evaluator decides whether to submit.
   - The reviewer compares the percentile to the target and routes the next round.
4. **`make dashboard`**: attempts across competitions, best CV / LB / percentile, experiments, CV vs LB, and an expandable timeline of every agent's reasoning.
5. **Evidence** for the runs reported below is in `runs/evidence/`: event logs, plans, experiments, submissions, plots and summaries.

## Results

| Competition | Metric | Best CV | Public LB | Leaderboard position | Iterations / submissions | Run |
|---|---|---|---|---|---|---|
| synthetic-binary (offline) | ROC AUC ↑ | 0.7735 | 0.7666 | top 6.5% of 400 synthetic teams | 1 / 1 | `runs/evidence/synthetic-binary` |
| synthetic-regression (offline, target top 5%) | RMSLE ↓ | 0.2681 | 0.2710 | top 10.0% of 400 synthetic teams | 4 / 1 | `runs/evidence/synthetic-regression` |
| titanic (LLM agents: Groq `openai/gpt-oss-120b`) | accuracy ↑ | 0.8395 (nested ensemble); 0.8373 best single | **0.78229** | top 22.2% of 11,466 (target 20%: not reached) | 4 / 2 | `runs/evidence/titanic-groq` |
| titanic, run 2 (after repeated CV + `cv_lcb` gate) | accuracy ↑ | 0.8339 (3×5-fold) | 0.76794 | top 69.8% | 4 / 1 | `runs/evidence/titanic-groq-run2` |
| titanic, run 3 (after the reviewer stall guard) | accuracy ↑ | 0.8395 (3×5-fold) | 0.76794 | top 69.8% | 4 / 1 | `runs/evidence/titanic-groq-run3` |
| titanic (heuristic agents, `--no-submit`) | accuracy ↑ | 0.8418 (nested ensemble); 0.8350 best single | | | 4 / 0 | `runs/evidence/titanic-heuristic` |
| house-prices-advanced-regression-techniques | RMSLE ↓ | not run | | | | |

**What the Titanic run shows.** The Planner inferred task, metric, target and id with no YAML overrides. The Data agent extracted passenger title, cabin deck and ticket prefix by regex and built family size and fare per person (the heuristic only takes the leading token). The Evaluator submitted the best single model rather than an ensemble that only mirrored it, then held back two rounds whose CV did not clear `min_improvement`. Its second submission, an ensemble with +0.2 points of CV and the highest fold std of the round, scored 0.77033 (top 65.7%): the public score covers all 418 test rows (0.78229 = 327/418, 0.77033 = 322/418), so that is 5 passengers, and the leaderboard is so dense around 0.775-0.78 that this moves the position by 40 points.

**Chasing the top 20%, and why it was not reached.** Top 20% on Titanic is 328/418 correct; the best run got 327. Two generic changes followed the first run: repeated CV with a pessimistic submit gate (`cv_lcb`), and a reviewer guardrail that forbids repeating a focus that just produced no CV gain (run 2 asked for `models` three times in a row with zero gain). Both worked as designed in runs 2 and 3: the gate held back every later candidate, and the guard switched `models` to `tuning` and forced a features round. Neither moved the leaderboard, because they target variance and the real limit is bias: across all three LLM runs CV sits at 0.83-0.84 and the public score at 0.768-0.782, a systematic 5.5-7 point gap that no choice between similar candidates can close, and the first submission of each run (which no gate can filter, since nothing is submitted yet) decides the score. Runs were capped at two after the first, by decision, rather than resubmitting variants until one crossed a one-passenger margin, which would measure leaderboard noise, not the agents. The next lever is the first submission itself: reserve it for the most regularised candidate, or let the reviewer trade CV for simplicity when the CV-LB gap is large.

**LLM vs heuristic.** Under the same nested CV, the differences (best single 0.8373 vs 0.8350, ensemble 0.8395 vs 0.8418) are well inside the fold std (~0.02). On a dataset this small, the LLM's measurable contribution is the quality and traceability of its choices, not CV. Reasoning errors seen in the logs, which guardrails do not catch: the reviewer read the CV-LB gap as overfitting and then asked for more capacity; the reviewer suggested CatBoost/XGBoost, which are outside the typed menu; the Evaluator read a higher in-sample ensemble CV as evidence *against* overfitting.

The synthetic leaderboard is generated, not real. Teams are spread between a constant prediction and the noise-free oracle. It exists to test the stopping and routing logic offline. Real positions come from the downloaded Kaggle leaderboard: rank / (teams + 1) at the time of submission.

## Design decisions and trade-offs

- **Typed action space instead of generated code.** The LLM's leverage is in *choices*: which text pattern to extract, which interactions to build, where to tune, when to stop. Generated-code agents can express more, but they fail in ways that are hard to sandbox and to reproduce.
- **Hard rules outside the LLM.** Submission budget, the improvement threshold, param bounds and column validity are enforced in code. The LLM's reasoning is logged next to what the guardrails changed, so overrides are visible.
- **CV is the primary signal; the LB is a check.** The evaluator submits only on CV improvement, which protects against public-LB overfitting and saves daily submissions.
- **Uncertainty-aware submissions.** Below 5,000 train rows CV is repeated 3 times with new shuffles (`cv_repeats`), because one 5-fold split is noisier than the differences between candidates (Titanic fold std ~0.02). Every candidate carries `cv_lcb`, its CV moved one standard error towards worse; the Evaluator ranks by it and the hard gate only allows a submission when `cv_lcb` beats the best submitted CV. This came from the first Titanic run, where an ensemble with +0.2 points of CV and the widest fold spread was submitted and lost 5 passengers on the leaderboard.
- **Greedy ensemble selection** (Caruana et al., 2004) on out-of-fold predictions. It is cheap and often beats the best single model. Its CV is nested (weights re-selected on k-1 folds of the OOF rows, scored on the held-out fold), because scoring it on the rows that chose its weights overstates it (Titanic accuracy: 0.846 in-sample vs 0.840 nested; the honest gain over the best single model is still +1.8 points). On pure-noise members the in-sample estimate reports a gain that does not exist, which the tests check. The in-sample number stays in `params.in_sample_cv`.
- **Fold-safe group target encoding.** Rows that share a key (a ticket, a booking, a household; a raw column or a regex-extracted part of one, such as a surname) often share the outcome. `group_target_encoding` gives each row the mean target of the *other* labelled members of its group, how many there are (0 = no information, unlike "all died") and the group size. `build_features` only assigns group ids; the statistics are computed inside `run_cv` for every fold from that fold's training rows, so a row's own label never enters its own group statistic (smoothing shrinks towards one fold-wide mean, never a per-row one, which would encode the row's label) and validation rows are encoded exactly like test rows. Controls: the same keys shuffled across rows (`scripts/make_noise_keys.py`, same cardinality and group sizes) give no CV gain, and a naive encoding that includes the row's own label gains 10+ points in the unit test, so the test can catch a leak. On Titanic with the LLM's own feature plan, encoding the ticket raises the bench's best `cv_lcb` from 0.8428 to 0.8512 (LightGBM 0.8474 → 0.8563), against 0.8367 on average for the shuffled ticket. The heuristic Data agent proposes it for high-cardinality text that repeats in train and reappears in test (`test_seen_pct` in the profile).
- **Heuristic parity.** Each agent's heuristic is a real baseline. It makes the system testable offline and gives a measurable bar for what the LLM adds: run the same config with `--llm mock` and with `--llm openai`, then compare.

## Limitations

- Tabular, single-target competitions with a CSV `train / test / sample_submission` layout. No images or text models, no multi-target, and no code competitions (notebook-only submission).
- Feature operations are a fixed menu. Domain-specific transformations outside it are not reachable.
- Large Playground datasets train on `sample_rows` for speed, which costs some accuracy.
- The percentile uses the public leaderboard at submission time. The private leaderboard can differ.
