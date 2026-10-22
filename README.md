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
| **Evaluator** | Picks a candidate and decides whether it deserves a submission. | `SubmitDecision` | The candidate is replaced by the simplest one the best-CV candidate does not beat (accuracy: one-sided McNemar on out-of-fold rows, p < 0.05; other metrics: within one standard error), so the LLM's pick can be overridden. No submission after the budget is spent, or unless `cv_lcb` beats the best submitted CV by `min_improvement` |
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

Leaderboard positions are where a **new** submission with that score lands on the 2026-10-22 Titanic leaderboard (11,499 teams). Kaggle ranks equal scores by who reached them first, so ties count as ahead: 327/418 shows as top 27.4%, top 20% needs **329/418** (328 lands at 22.2%). `summary.json` files written before this convention was fixed report the optimistic one (327 = 22.2%).

| Competition | Metric | CV of submitted (best CV) | Public | Position | Iterations / submissions | Run |
|---|---|---|---|---|---|---|
| synthetic-binary (offline, heuristic agents) | ROC AUC ↑ | 0.7735 | 0.7666 | top 6.5% of 400 synthetic teams | 1 / 1 | `runs/evidence/synthetic-binary` |
| synthetic-regression (offline, heuristic agents, target top 5%) | RMSLE ↓ | 0.2681 | 0.2710 | top 10.0% of 400 synthetic teams | 4 / 1 | `runs/evidence/synthetic-regression` |
| titanic, run 1 (LLM agents: Groq `openai/gpt-oss-120b`) | accuracy ↑ | 0.8373 (0.8395) | **327/418** | top 27.4% | 4 / 2 (2nd: 322) | `runs/evidence/titanic-groq` |
| titanic, run 2 (repeated CV + `cv_lcb` gate) | accuracy ↑ | 0.8339 | 321/418 | top 73.3% | 4 / 1 | `runs/evidence/titanic-groq-run2` |
| titanic, run 3 (reviewer stall guard) | accuracy ↑ | 0.8395 | 321/418 | top 73.3% | 4 / 1 | `runs/evidence/titanic-groq-run3` |
| titanic, run 4 (group target encoding; LLM chose category keys) | accuracy ↑ | 0.8272 | 311/418 | top 93.6% | 3 / 1 | `runs/evidence/titanic-groq-run4` |
| titanic, run 5 (category keys skipped, heuristic keys used) | accuracy ↑ | 0.8418 | 319/418 | top 84.2% | 1 / 1 | `runs/evidence/titanic-groq-run5` |
| titanic, run 6 (`group_override` + one-SE selection) | accuracy ↑ | 0.8361 (0.8429) | 311/418 | top 93.6% | 4 / 1 (2nd hit the daily limit) | `runs/evidence/titanic-groq-run6` |
| titanic (heuristic agents, `--no-submit`) | accuracy ↑ | (0.8418) | | | 4 / 0 | `runs/evidence/titanic-heuristic` |
| house-prices-advanced-regression-techniques | RMSLE ↓ | not run | | | | |

**Target: not reached by the agent team.** Best agent result: 327/418 (top 27.4%). Nothing used leaked labels or multiple accounts.

**Human involvement, in the order it happened.** The Titanic work was not hands-off, and the later changes were made knowing public scores:
1. After runs 1-3 (public 321-327), a multi-agent workflow researched leak-free Titanic techniques, implemented three levers in isolated worktrees and had adversarial verifiers reproduce them; only fold-safe group target encoding was kept (+0.6 to +0.7 `cv_lcb` over 5 CV seeds). A one-standard-error selection rule was rejected then (across model families it lost 1.24 ± 0.49 points against `cv_lcb` on 20 held-out splits).
2. Runs 4-5 showed the LLM using category keys (ticket prefix, cabin letter) for the encoding, which lets a model read training labels back (CV 0.847 → 0.827, public 311). Keys whose repeated groups average more than 10 rows are now skipped; the threshold of 10 was set from run 4's keys (12 and 42).
3. After all seven agent submissions, three manual baselines were submitted to test why the agents stall (below). Their definitions were fixed before their own scores were seen, but the woman-child-group (WCG) rule is a published notebook (C. Deotte) known for its public score.
4. After the WCG baseline scored 334, `group_override` was written in the shape of that rule (a shallow tree changed only where a row's group is unanimous), its defaults chosen on Titanic CV, and the orchestrator made to always evaluate it. After run 6 (311), category handling was changed (CART ordering, rare levels to the prior) and the Evaluator rule was changed to "the simplest candidate the best-CV candidate does not beat with a one-sided McNemar test, p < 0.05, on the same out-of-fold rows". **That rule was adopted after seeing that, on a local run, it selects `group_override`.** An adversarial review found no label leakage (flipped validation labels change no prediction; shuffled labels give CV 0.605-0.614 against a majority rate of 0.616), and found that the test leans towards complex candidates when the lead is the best of ~20 estimates (P(p < 0.05) = 0.22 instead of 0.05 under the null).

Any Titanic score submitted after step 4 measures this author-designed pipeline, not an autonomous discovery by the agents, and is not independent confirmation of the rule.

**Why the agent team stops near 0.77: what was tested.**

| Model | CV (3×5) | Public | Position |
|---|---|---|---|
| gender rule (no fitted parameter) | 0.7868 | 320/418 | top 81.4% |
| shallow random forest, 6 features | 0.8358 | 324/418 | top 61.4% |
| agent team, 7 submissions | 0.827-0.842 | 311-327 | top 27.4-93.6% |
| WCG rule (gender + 16 unanimous-group overrides, train labels only) | 0.8417 | **334/418** | **top 8.7%** |

- CV does not rank these models: the WCG rule and the agents' best-CV model have the same CV (0.8417 vs 0.8418) and differ by 15 passengers on the public test (55 differing rows, 35-20, two-sided sign test p = 0.058; against the agents' best public file, 327, p = 0.38). This is consistent with a model-class effect that CV cannot see, but does not by itself establish one. Across the agents' 7 submissions, Spearman ρ(CV, public) = 0.36 (p = 0.43).
- The CV-public gap is 2.1 points for the gender rule, 4.3 for WCG, 6.1 for the 6-feature forest and 5.5-9.2 for the agents' 7 submissions.
- Not leakage from random folds: a validation row shares a ticket with training rows 34% of the time, a test row 36%; grouping folds by ticket costs 0.6 points of CV.
- Not drift: adversarial validation AUC train vs test = 0.51.
- Leaderboard noise: one binomial standard error is 8.5 passengers; 327 is not distinguishable from 319-322 (sign test on the differing rows, p = 0.23-0.38), only from 311 (p = 0.014).
- `group_override` after the review fixes: CV 0.830 on average over the LLM's own feature plans and two CV seeds (minimum 0.827) against 0.8417 for the hand-specified rule; its test predictions differ from the WCG file in 20 rows, which bounds their public score to 314-346 without submitting. With these fixes, a local run (heuristic agents, no submission) no longer selects it: LightGBM (CV 0.847) beats it on the paired test, so the Evaluator picks boosted trees again, whose files bound to 299-349.

**LLM vs heuristic.** Under the same nested CV, the differences (best single 0.8373 vs 0.8350, ensemble 0.8395 vs 0.8418) are well inside the fold std (~0.02). On a dataset this small, the LLM's measurable contribution is the quality and traceability of its choices, not CV. Reasoning errors seen in the logs, which guardrails do not catch: the reviewer read the CV-LB gap as overfitting and then asked for more capacity; the reviewer suggested CatBoost/XGBoost, which are outside the typed menu; the Evaluator read a higher in-sample ensemble CV as evidence *against* overfitting.

The synthetic leaderboard is generated, not real. Teams are spread between a constant prediction and the noise-free oracle. It exists to test the stopping and routing logic offline. Real positions come from the downloaded Kaggle leaderboard: rank / (teams + 1) at the time of submission.

## Design decisions and trade-offs

- **Typed action space instead of generated code.** The LLM's leverage is in *choices*: which text pattern to extract, which interactions to build, where to tune, when to stop. Generated-code agents can express more, but they fail in ways that are hard to sandbox and to reproduce.
- **Hard rules outside the LLM.** Submission budget, the improvement threshold, param bounds and column validity are enforced in code. The LLM's reasoning is logged next to what the guardrails changed, so overrides are visible.
- **CV is the primary signal; the LB is a check.** The evaluator submits only on CV improvement, which protects against public-LB overfitting and saves daily submissions.
- **Uncertainty-aware submissions.** Below 5,000 train rows CV is repeated 3 times with new shuffles (`cv_repeats`), because one 5-fold split is noisier than the differences between candidates (Titanic fold std ~0.02). Every candidate carries `cv_lcb`, its CV moved one standard error towards worse; the hard gate only allows a submission when `cv_lcb` beats the best submitted CV. This came from the first Titanic run, where an ensemble with +0.2 points of CV and the widest fold spread was submitted and lost 5 passengers on the leaderboard.
- **Greedy ensemble selection** (Caruana et al., 2004) on out-of-fold predictions. It is cheap and often beats the best single model. Its CV is nested (weights re-selected on k-1 folds of the OOF rows, scored on the held-out fold), because scoring it on the rows that chose its weights overstates it (Titanic accuracy: 0.846 in-sample vs 0.840 nested; the honest gain over the best single model is still +1.8 points). On pure-noise members the in-sample estimate reports a gain that does not exist, which the tests check. The in-sample number stays in `params.in_sample_cv`.
- **Fold-safe group target encoding.** Rows that share a key (a ticket, a booking, a household; a raw column or a regex-extracted part of one, such as a surname) often share the outcome. `group_target_encoding` gives each row the mean target of the *other* labelled members of its group, how many there are (0 = no information, unlike "all died") and the group size. `build_features` only assigns group ids; the statistics are computed inside `run_cv` for every fold from that fold's training rows, so a row's own label never enters its own group statistic (smoothing shrinks towards one fold-wide mean, never a per-row one, which would encode the row's label) and validation rows are encoded exactly like test rows. Controls: the same keys shuffled across rows (`scripts/make_noise_keys.py`, same cardinality and group sizes) give no CV gain, and a naive encoding that includes the row's own label gains 10+ points in the unit test, so the test can catch a leak. On Titanic with the LLM's own feature plan, encoding the ticket raises the bench's best `cv_lcb` by +0.6 to +0.7 points on average over 5 CV seeds, positive on all 5 (+0.84 at seed 42: 0.8428 → 0.8512; the ticket key was also the best of 8 variants compared, so read the top number as optimistic), against a loss for the shuffled ticket (0.8367 on average at seed 42). The heuristic Data agent proposes it for high-cardinality text that repeats in train and reappears in test (`test_seen_pct` in the profile).
- **Heuristic parity.** Each agent's heuristic is a real baseline. It makes the system testable offline and gives a measurable bar for what the LLM adds: run the same config with `--llm mock` and with `--llm openai`, then compare.

## Limitations

- Tabular, single-target competitions with a CSV `train / test / sample_submission` layout. No images or text models, no multi-target, and no code competitions (notebook-only submission).
- Feature operations are a fixed menu. Domain-specific transformations outside it are not reachable.
- Large Playground datasets train on `sample_rows` for speed, which costs some accuracy.
- The percentile uses the public leaderboard at submission time. The private leaderboard can differ.
