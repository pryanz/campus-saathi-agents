# Campus Saathi

Campus Saathi is a campus-assistance application for students and parents completing physical reporting and related college procedures. It combines:

- a campus routing simulator with classical search, heuristic search, reinforcement learning, value iteration, and simulated annealing; and
- a grounded knowledge assistant that retrieves records from `data/knowledge/` and uses Groq for structured answer generation.

The application is designed to avoid inventing campus facts. Retrieval has a score threshold (the hard gate); unsupported questions return a safe `not_available` response with an official-site fallback.

## Routing problem

A student starts at the gate and must complete every reporting step. The campus is a weighted graph of 11 nodes in neutral, academic, and hostel clusters. Each procedure has a room, mean service time, and counter configuration in `data/steps.json`.

The simulator state is `(current_room, completed_steps)`. Actions are `("move", neighbour)` and `("complete", current_room)`. The goal is to complete every configured step, with:

```text
total cost = travel time + queue waiting time + service time
```

The formal problem definition is in [`docs/PROBLEM.md`](docs/PROBLEM.md), with environment and PEAS descriptions in [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) and [`docs/PEAS.md`](docs/PEAS.md).

### Agents

| Agent | Family | Implementation | Notes |
| --- | --- | --- | --- |
| BFS / DFS / IDS | Uninformed search | `algorithms/` | Baseline graph searches |
| UCS | Cost-aware search | `algorithms/ucs.py` | Uniform-cost search; A* with `h = 0` |
| A* | Informed search | `algorithms/astar.py` | Farthest-remaining-room and cluster-aware heuristics |
| Greedy Nearest | Heuristic | `algorithms/greedy_nearest.py` | Chooses the nearest unfinished room |
| Simulated Annealing | Local search | `algorithms/simulated_annealing.py` | Optimizes the visit order |
| Value Iteration | MDP planning | `algorithms/value_iteration.py` | Includes low/medium/high congestion states |
| Q-Learning | Reinforcement learning | `agents/q_learning_agent.py` | Tabular epsilon-greedy agent trained online |

All agents implement the common `BaseAgent` interface. `CampusSimulator` uses seeded randomness, 20 background students, FIFO room queues, and congestion-dependent waiting. Set `congestion=false` to compare travel and service costs without queue dynamics. API simulations use seed `42`; Q-learning tables are cached by iteration count after the first request.

## Repository layout

| Path | Purpose |
| --- | --- |
| `backend/` | FastAPI API, intent routing, BM25 retrieval, structured RAG, and Groq integration |
| `agents/`, `algorithms/`, `core/`, `envi/` | Search agents, planning algorithms, metrics, campus graph, and simulator |
| `data/` | Campus graph, procedure steps, and typed knowledge records |
| `frontend/` | React/Vite interface with map, chat, and metrics panels |
| `tests/` | Unit, contract, retrieval, algorithm, and evaluation-set tests |
| `scripts/` | Knowledge and evaluation-set validators |
| `docs/` | API contract, environment, problem, PEAS, conventions, and algorithm notes |
| `experiments/` | Training/evaluation entry points for selected agents |

## Requirements

- Python 3.14 is the currently verified interpreter in this workspace.
- Node.js and npm for the frontend.
- A Groq API key for requests that reach the LLM. Greeting, link, and most local routing behavior can be tested without an LLM, but importing `backend.groq_service` still requires `GROQ_API_KEY`.

## Setup

Create a virtual environment and install the pinned backend dependencies:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Copy the environment examples and fill in the backend key:

```bash
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
```

Set at least `GROQ_API_KEY` in `backend/.env`. The other backend settings are:

| Variable | Default | Meaning |
| --- | --- | --- |
| `GROQ_API_KEY` | none | API key required by the Groq client |
| `GROQ_MODEL` | `qwen/qwen3.8-27b` | Groq model used for navigation and structured RAG |
| `KB_MIN_SCORE` | `0.20` | BM25 retrieval threshold used by the knowledge hard gate |
| `CORS_ORIGINS` | not currently read by the backend | Documented example value; the API currently permits all origins |
| `GROQ_METRICS_LOG` | `logs/groq_calls.jsonl` | Redacted JSONL timing and usage log for each Groq call |

Do not commit `backend/.env` or API keys.

## Run locally

Start the API from the repository root:

```bash
.venv/bin/uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal, start the frontend:

```bash
cd frontend
npm install
npm run dev
```

The Vite app uses `VITE_API_URL`, defaulting to `http://127.0.0.1:8000`.

## API

### `GET /api/map`

Returns the graph in `data/campus_map.json`.

### `GET /api/simulate/{agent_type}`

Runs one campus episode. Supported agent names are `a_star`, `bfs`, `dfs`, `greedy_nearest`, `ids`, `simulated_annealing`, `q_learning`, `value_iteration`, and `ucs`.

Useful query parameters include:

```text
congestion=true|false
iterations=100                 # used by q_learning
start_location=gate
completed_steps=gate&completed_steps=shop
```

The response includes the route trace and `metrics` with `travel_time`, `wait_time`, `service_time`, and `total_cost`.

### `POST /api/chat`

Request:

```json
{
  "query": "Where is the central library?",
  "history": []
}
```

`history` accepts at most six non-empty user/assistant messages. The response contains `response`, `found`, `answer_type`, `sources`, `links`, and optional `node_id`/`last_verified` fields. `answer_type` is one of `knowledge`, `navigation`, `link`, or `not_available`.

See [`docs/CHAT_API.md`](docs/CHAT_API.md) for the contract. The API currently enables CORS for all origins; restrict this before production deployment.

## Knowledge data and evaluation set

Knowledge records are JSON files under `data/knowledge/`. Their schema and cross-file conventions are documented in [`data/knowledge/SCHEMA.md`](data/knowledge/SCHEMA.md) and [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md).

Run the data checks from the repository root:

```bash
.venv/bin/python scripts/validate_knowledge.py
.venv/bin/python scripts/check_eval_set.py
.venv/bin/python scripts/run_eval.py --output .benchmarks/eval.json
```

The evaluation set is intended to cover answerable, unanswerable, near-miss, prompt-injection, and Hinglish queries. It currently contains 80 questions and checks against the knowledge-record IDs.

## Tests and frontend checks

```bash
.venv/bin/pytest -q
cd frontend
npm run build
npm run lint
```

The frontend build is a production compile. `npm run lint` currently completes with warnings in `App.jsx` and `CampusMap.jsx`; these should be cleaned up before treating lint as a quality gate.

## Benchmark plan

`scripts/run_eval.py` runs the checked question set and records per-question responses and latency. The following additional measurements make results reproducible and useful:

| Area | Metrics | Recommended comparison |
| --- | --- | --- |
| LLM serving | input tokens/sec, output tokens/sec, time-to-first-token, end-to-end latency, p50/p95, errors, retries | model and prompt variants under the same query set |
| RAG | retrieval latency, top-score distribution, recall@k, answerable precision/recall, unsupported-question false-positive rate | `KB_MIN_SCORE` values and BM25 top-k settings |
| Routing | total cost, travel/wait/service time, nodes expanded, route success, optimality gap | every agent with congestion on/off and fixed seeds |
| API | p50/p95 latency, throughput, status codes, concurrency, payload size | `/api/chat`, `/api/simulate/{agent_type}`, and `/api/map` separately |
| Frontend | bundle size, initial load, interaction latency | production build and representative device/network profiles |

For token throughput, use provider-reported usage rather than estimating from characters:

```text
output_tokens_per_second = completion_tokens / generation_seconds
input_tokens_per_second  = prompt_tokens / prompt_processing_seconds
```

Each Groq call now writes latency, `response.usage.prompt_tokens`, `completion_tokens`, `total_tokens`, and derived output tokens/sec to the JSONL metrics log. Prompts, responses, API keys, and personal data are not written. Calls are non-streaming, so time-to-first-token is not available; use end-to-end call latency. For fair results, warm up the process, fix the model and seed, run at least 30 repetitions per case, report p50/p95, and separate first-request latency from warm-cache latency.

A useful first benchmark corpus is `tests/eval/questions.json`, supplemented with navigation queries that specify start locations and completed steps. Report answer quality separately from speed: an answer can be fast but fail grounding or the hard-gate safety requirement.

## Known audit notes

- The repository's knowledge validator and evaluation-set checker pass against the current data.
- A live 80-question run recorded 0/25 unsupported false positives (0.0%); 24/55 supported questions produced a grounded answer. The latter is limited by an eval-data mismatch: expected `procedure_*` IDs come from `data/steps.json`, while the RAG index loads only `data/knowledge/*.json`.
- That run made 70 Groq calls including retries: 34,173 prompt tokens, 3,997 completion tokens, p50 latency 3.55s, p95 latency 10.37s, and mean output throughput 72.34 tokens/sec. These are one-run results, not a service-level guarantee.
- The frontend production build succeeds; lint exits successfully but emits warnings.
- The Python suite requires the dependencies in `requirements.txt`. Under Python 3.14, the dependency-complete run progresses through the algorithm tests but stalls at `tests/test_chat_contract.py::test_chat_endpoint_basic` because Starlette's synchronous `TestClient`/AnyIO portal hangs even for a minimal app; use Python 3.13 or migrate these tests to an async ASGI client.
- `backend/.env.example` includes `CORS_ORIGINS`, but the backend does not currently read it and configures `allow_origins=["*"]`.
- `docs/CHAT_API.md` contains Markdown-style links inside JSON examples; actual API URLs are plain URL strings as defined by the Pydantic response models.
- `TestClient` stalls under the verified Python 3.14 environment even for a minimal FastAPI app. The direct endpoint function succeeds; the failure is in the Starlette/AnyIO test harness compatibility and is not a production endpoint hang. `httpx2` is now declared because Starlette 1.10 requires it, but the Python 3.14 harness issue remains to be resolved or tested under Python 3.13.

## License

See [`LICENSE`](LICENSE).
