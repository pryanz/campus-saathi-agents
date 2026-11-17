
import os
import json
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from backend.config import GROQ_MODEL
from backend.observability import groq_call


# ============================================================
# ENVIRONMENT AND GROQ CLIENT
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
ENV_PATH = CURRENT_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY is missing. Please configure it in backend/.env."
    )

client = Groq(api_key=api_key)


# ============================================================
# SIMULATION CONFIGURATION
# ============================================================

ALGORITHMS = [
    "a_star",
    "bfs",
    "dfs",
    "greedy_nearest",
    "q_learning",
]


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "simulate_all_paths",
            "description": (
    "Run simulations across all available pathfinding "
    "algorithms and return their respective metrics and trajectories."
),            "parameters": {
                "type": "object",
                "properties": {
                    "current_location": {
                        "type": "string",
                        "description": (
                            "User current location or starting room "
                            "(defaults to 'gate')."
                        ),
                    },
                    "completed_steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Rooms already completed "
                            "(defaults to ['gate'])."
                        ),
                    },
                    "congestion": {
                        "type": "boolean",
                        "description": (
                            "Whether to consider dynamic congestion "
                            "(default true)."
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_path",
            "description": (
                "Run simulation for a single specific pathfinding algorithm."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "algorithm": {
                        "type": "string",
                        "enum": ALGORITHMS,
                        "description": "The pathfinding algorithm to simulate.",
                    },
                    "current_location": {
                        "type": "string",
                        "description": (
                            "User current location or starting room "
                            "(defaults to 'gate')."
                        ),
                    },
                    "completed_steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Rooms already completed "
                            "(defaults to ['gate'])."
                        ),
                    },
                    "congestion": {
                        "type": "boolean",
                        "description": (
                            "Whether to consider dynamic congestion "
                            "(default true)."
                        ),
                    },
                },
                "required": ["algorithm"],
            },
        },
    },
]


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """You are Campus Saathi, an intelligent multi-agent
campus pathfinding assistant.

Your goal is to guide students and visitors through campus procedures
along the most optimal path.

Instructions:

1. Extract the user's current location and completed steps from their query.

2. Fallback logic:
   - If no start location is specified, assume "gate".
   - If a location is specified, use it.
   - Always assume "gate" as completed by default if no completed
     steps are provided.

3. Call simulate_all_paths (or simulate_path across all algorithms)
   to fetch the pathfinding metrics.

4. Internally evaluate all algorithm results to select the optimal
   path having the lowest total cost and travel time.

5. In your final response, provide ONLY:
   - The optimal path (route sequence of locations to visit).
   - The immediate next step the user must take right now.

Strictly DO NOT mention algorithm names (such as A*, Greedy, BFS,
DFS, Q-Learning) or numerical metrics (travel time, wait time,
cost, steps) unless the user explicitly asks for them.
"""


# ============================================================
# SIMULATION RESULT SUMMARIZATION
# ============================================================

def summarize_simulation_result(result):
    """Extract useful route information from a simulation result."""

    if not isinstance(result, dict):
        return result

    trace = result.get("trace", [])

    completed_sequence = [
        step["current_room"]
        for step in trace
        if step.get("action_taken", [None])[0] == "complete"
    ]

    full_moves = [
        step["current_room"]
        for step in trace
        if step.get("action_taken", [None])[0] == "move"
    ]

    return {
        "agent": result.get("agent"),
        "total_steps": result.get("total_steps"),
        "metrics": result.get("metrics"),
        "route": full_moves,
        "completed_sequence": completed_sequence,
    }


# ============================================================
# CAMPUS SAATHI CHAT HELPER
# ============================================================

def query_campus_helper(user_query: str, run_simulation_callback):
    """Generate a campus response using Groq and simulation tools."""

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_query,
        },
    ]

    # First Groq request: decide whether simulation tools are required.
    response = groq_call(
        "navigation_tool_selection",
        lambda: client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        ),
        model=GROQ_MODEL,
    )

    response_message = response.choices[0].message

    if not response_message.tool_calls:
        return response_message.content or (
            "I'm sorry, I couldn't generate a response."
        )

    messages.append(response_message)

    # Execute all requested simulation tool calls.
    for tool_call in response_message.tool_calls:

        function_args = (
            json.loads(tool_call.function.arguments)
            if tool_call.function.arguments
            else {}
        )

        congestion = function_args.get("congestion", True)
        location = function_args.get("current_location", "gate")
        completed = function_args.get("completed_steps", ["gate"])

        if tool_call.function.name == "simulate_all_paths":

            simulation_results = {}

            for algorithm in ALGORITHMS:
                raw_result = run_simulation_callback(
                    agent_type=algorithm,
                    congestion=congestion,
                    start_location=location,
                    completed_steps=completed,
                )

                simulation_results[algorithm] = (
                    summarize_simulation_result(raw_result)
                )

        elif tool_call.function.name == "simulate_path":

            algorithm = function_args.get("algorithm", "a_star")

            raw_result = run_simulation_callback(
                agent_type=algorithm,
                congestion=congestion,
                start_location=location,
                completed_steps=completed,
            )

            simulation_results = summarize_simulation_result(raw_result)

        else:
            simulation_results = {
                "error": "Unsupported simulation tool."
            }

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": json.dumps(simulation_results),
            }
        )

    # Second Groq request: generate the final response
    # after all simulation tool calls have completed.
    final_response = groq_call(
        "navigation_final_answer",
        lambda: client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
        ),
        model=GROQ_MODEL,
    )

    return (
        final_response.choices[0].message.content
        or "I'm sorry, I couldn't generate a response."
    )


def generate_structured_rag_response(
    system_prompt: str,
    user_message: str,
    history: list[dict] | None = None,
    temperature: float = 0.1,
) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    response = groq_call(
        "knowledge_structured_answer",
        lambda: client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        ),
        model=GROQ_MODEL,
    )
    return response.choices[0].message.content or ""
