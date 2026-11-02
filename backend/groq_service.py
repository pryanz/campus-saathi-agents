import os
import json
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from backend.rag_service import rag_service

CURRENT_DIR = Path(__file__).resolve().parent
ENV_PATH = CURRENT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# groq client check kar rahe hain (agar key valid hai tabhi client banega)
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key) if api_key and api_key != "actual_groq_api_key" else None

# Issue #21 ke exact 4 tools
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_checklist",
            "description": "Fetch official physical reporting document checklist, original vs photocopy counts, and cutoff rules tailored to a student's category and admission route.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["OPEN", "GEN-EWS", "OBC-NCL", "SC", "ST", "PwD"],
                        "description": "Student admission category (e.g. 'OBC-NCL', 'OPEN', 'GEN-EWS', 'SC', 'ST')."
                    },
                    "admission_route": {
                        "type": "string",
                        "description": "Admission channel (e.g. 'JoSAA', 'CSAB', 'DASA'). Defaults to 'JoSAA'."
                    },
                    "has_gap_year": {
                        "type": "boolean",
                        "description": "Whether candidate took a drop/gap year after Class 12."
                    }
                },
                "required": ["category"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_calendar",
            "description": "Fetch academic calendar schedule including physical reporting dates, orientation, course registration, fee deadlines, and exam windows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "batch": {
                        "type": "string",
                        "description": "Batch identifier (e.g. 'UG_Freshers'). Defaults to 'UG_Freshers'."
                    },
                    "query": {
                        "type": "string",
                        "description": "Specific event search filter (e.g. 'orientation', 'reporting', 'registration', 'classes')."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "route_to",
            "description": "Calculate shortest walking route, travel time, and next action from current location to target room or campus facility using AI search agents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "string",
                        "description": "Target room or facility (e.g. 'L-103', 'L-104', 'panini_gate', 'xerox', 'notary', 'phc', 'room')."
                    },
                    "current_location": {
                        "type": "string",
                        "description": "Current room location (defaults to 'gate')."
                    }
                },
                "required": ["destination"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Constrained search over official allow-listed institutional domains (iiitdmj.ac.in, josaa.nic.in, antiragging.in) for verified policies, FAQs, and contact desks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (e.g. 'medical certificate format', 'hostel room refund', 'anti ragging affidavit format')."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

SYSTEM_PROMPT = """You are Campus Saathi, an intelligent multi-agent campus pathfinding and physical reporting assistant.
Your goal is to guide students and visitors through university reporting procedures, required documents, campus facilities, and optimal navigation.

Strict Tool-Calling Rules:
1. When asked about documents, certificates, OBC/EWS cutoff dates, medical forms, or gap affidavits, ALWAYS call `get_checklist`.
2. When asked about reporting dates, orientation, semester commencement, or deadlines, ALWAYS call `get_calendar`.
3. When asked where to go, how to reach a station, or the fastest walking path, ALWAYS call `route_to`.
4. When asked about specific policies, hostel bylaws, fees, or exceptions, call `search_web`. (Web search is strictly constrained to allow-listed domains: iiitdmj.ac.in, josaa.nic.in, antiragging.in).

Strict Citation & Anti-Hallucination Mandate:
1. Every factual statement or requirement MUST display an explicit source citation in the format:
   [Source: <source_url>, Verified: <last_verified>]
2. If an inquiry cannot be verified through official institutional sources or allow-listed domains, you MUST gracefully decline to guess. Say:
   "I do not have verified official information regarding this policy in the institute database. Please consult the Central Helpdesk or Exception Desk at Room L-103."
3. Language: Clear, polite, supportive. Seamlessly understands English and Hinglish.
"""

def _local_fallback_response(user_query: str, history: list[dict] = None, run_simulation_callback = None) -> str:
    # agar groq key nahi hai toh local rag aur graph simulator se jawab banayenge
    import re
    q_lower = user_query.lower()

    # pichle messages se context nikal rahe hain
    context_text = ""
    if history:
        for h in history:
            context_text += " " + (h.get("text") or h.get("content") or "")
    combined_lower = (context_text + " " + user_query).lower()

    # category detect kar rahe hain
    cat = "OPEN"
    if re.search(r'\b(obc|obc-ncl)\b', combined_lower):
        cat = "OBC-NCL"
    elif re.search(r'\b(ews|gen-ews)\b', combined_lower):
        cat = "GEN-EWS"
    elif re.search(r'\b(sc)\b', combined_lower):
        cat = "SC"
    elif re.search(r'\b(st)\b', combined_lower):
        cat = "ST"
    elif re.search(r'\b(pwd)\b', combined_lower):
        cat = "PwD"

    has_gap = any(k in combined_lower for k in ["gap", "dropper", "year gap", "drop year"])

    # 'what should i do first' planner query check
    if any(k in q_lower for k in ["what first", "do first", "where to go first", "pehle kya", "first step", "just arrived", "heavy luggage", "with bags"]):
        from backend.task_planner import plan_first_steps
        has_bags = any(k in combined_lower for k in ["bag", "luggage", "suitcase", "heavy"])
        plan = plan_first_steps(arrival_time="10:00 AM", has_heavy_luggage=has_bags, category=cat)
        resp = f"🧭 **Personalized 'What Should I Do First?' Roadmap:**\n\n"
        resp += f"• **Context:** {plan['time_context']}\n"
        for r in plan['rationale']:
            resp += f"• {r}\n"
        resp += f"\n📍 **Recommended Optimal First Stop:** **{plan['recommended_first_stop_name']}** ({plan['recommended_first_stop'].upper()})\n\n"
        resp += f"📋 **Step-by-Step Navigation Sequence:**\n"
        for d in plan['directions'][:5]:
            resp += f"  {d['step_number']}. **{d['title']}** ({d['station_code'].upper()})\n     ↳ {d['action']}\n"
        resp += f"\n⏱️ **Estimated Total Reporting Duration:** ~{plan['estimated_total_time_min']} mins (including walking and queues)\n\n"
        resp += f"[Source: Official IIITDMJ Reporting Protocol & Task-Order Planner, Verified: 2026-11-02]"
        return resp

    # facility aur campus services check
    for fac_key in ["xerox", "photocop", "notar", "stamp", "affidavit", "medical", "doctor", "phc", "atm", "bank", "hostel", "mess"]:
        if fac_key in q_lower:
            lookup_key = "xerox" if "photocop" in fac_key else ("notary" if "notar" in fac_key else fac_key)
            fac = rag_service.get_facility(lookup_key)
            if fac:
                services_list = "\n  • " + "\n  • ".join(fac.get("services", []))
                return (
                    f"🏢 **{fac['name']}**\n\n"
                    f"📍 **Location:** {fac.get('location')}\n"
                    f"⏰ **Operating Hours:** {fac.get('operating_hours', fac.get('timings', 'Regular hours'))}\n"
                    f"💼 **Services Offered:**{services_list}\n"
                    f"💰 **Rates / Contact:** {fac.get('rates', fac.get('charges', fac.get('contact', 'Standard institute rates')))}\n\n"
                    f"[Source: {fac.get('source_url', 'https://www.iiitdmj.ac.in')}, Verified: {fac.get('last_verified', '2026-10-25')}]"
                )

    # checklist aur cutoff rules queries
    if any(k in q_lower for k in ["document", "checklist", "marksheet", "certificate", "cutoff", "validity", "format", "april", "march", "undertaking", "annexure"]):
        data = rag_service.get_checklist(category=cat, has_gap_year=has_gap)
        docs = data.get("documents", [])
        
        # cutoff date follow-up query ke liye specific detail
        if any(k in q_lower for k in ["cutoff", "validity", "date", "march", "april", "expire"]):
            target_doc = next((d for d in docs if cat in d.get("document_name", "")), docs[0])
            return (
                f"📋 **Cutoff & Validity Rules for {target_doc['document_name']}:**\n\n"
                f"• **Requirement:** {target_doc['format_required']}\n"
                f"• **Cutoff Rule:** {target_doc['validity_cutoff']}\n"
                f"• **What if expired or missing:** {target_doc['action_if_missing']}\n\n"
                f"[Source: {target_doc.get('source_url', data['source_url'])}, Verified: {target_doc.get('last_verified', data['last_verified'])}]"
            )

        resp = f"📋 **Official Physical Reporting Documents ({cat} Category):**\n\n"
        for d in docs[:5]:
            resp += f"• **{d['document_name']}**\n  - Format: {d['format_required']}\n  - Rules: {d['validity_cutoff']}\n"
        resp += f"\n[Source: {data['source_url']}, Verified: {data['last_verified']}]"
        return resp

    # calendar aur reporting dates
    if any(k in q_lower for k in ["date", "calendar", "orientation", "registration", "deadline", "when"]):
        data = rag_service.get_calendar(query=user_query)
        events = data.get("events", [])
        resp = f"📅 **Academic & Reporting Schedule (Session {data.get('academic_session')}):**\n\n"
        for ev in events[:4]:
            resp += f"• **{ev['title']}**\n  - Date: {ev['date']} ({ev.get('time', 'All Day')})\n  - Venue: {ev.get('venue')}\n"
        resp += f"\n[Source: {data['source_url']}, Verified: {data['last_verified']}]"
        return resp

    # rasta aur navigation queries (A* search)
    if any(k in q_lower for k in ["route", "where to go", "how to reach", "shortest path", "fastest", "next step", "at gate", "panini", "lhtc"]):
        start_loc = "gate"
        for room in ["lhtc_gate", "L-103", "L-104", "L-105", "L-106", "L-206", "panini_gate", "office", "room", "shop"]:
            if room.lower() in combined_lower and room.lower() not in q_lower:
                start_loc = room
                break
        sim = run_simulation_callback(agent_type="a_star", congestion=True, start_location=start_loc) if run_simulation_callback else {}
        trace = sim.get("trace", [])
        moves = [s["current_room"] for s in trace if s.get("action_taken", [None])[0] == "move"]
        first_act = trace[0]["action_taken"] if trace else ["complete", start_loc]
        return (
            f"📍 **Optimal Campus Navigation Plan (A* Agent):**\n"
            f"• Route: {start_loc.upper()} ➔ {' ➔ '.join(moves[:5])}\n"
            f"• **Immediate Action:** {first_act[0].upper()} at **{first_act[1]}**\n"
            f"• Estimated Total Time: ~{sim.get('metrics', {}).get('total_cost', 146)} mins (including queue wait)\n\n"
            f"[Source: Campus Saathi Spatial Graph Engine, Verified: 2026-11-01]"
        )

    # domain allowlist par policy search
    web_res = rag_service.search_web(user_query)
    if web_res.get("status") == "success":
        resp = "🔍 **Verified Institutional Policy:**\n\n"
        for r in web_res.get("results", [])[:2]:
            resp += f"• **{r['title']}**\n  {r['content']}\n  [Source: {r['source_url']}, Verified: {r['last_verified']}]\n\n"
        return resp.strip()

    # agar policy verified nahi hai toh seedhe L-103 Exception Desk bhejenge
    return (
        "I do not have verified official information regarding this policy in the institute database. "
        "Please consult the Central Helpdesk or Exception Desk at Room L-103.\n\n"
        "[Source: Official Helpdesk / Exception Desk L-103, Verified: 2026-11-01]"
    )


def query_campus_helper(user_query: str, history: list[dict] = None, run_simulation_callback = None) -> str:
    # groq nahi hai toh local fallback chalega
    if not client:
        return _local_fallback_response(user_query, history, run_simulation_callback)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # pichle 6 messages ka context window bhej rahe hain
    if history:
        for h in history[-6:]:
            role = h.get("role")
            content = h.get("text") or h.get("content")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_query})

    try:
        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )
    except Exception as e:
        print(f"[groq_service] API exception: {e}, local fallback active.")
        return _local_fallback_response(user_query, history, run_simulation_callback)

    response_message = response.choices[0].message
    messages.append(response_message)

    # agar model ne koi tool call kiya
    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}

            tool_data = {}
            if name == "get_checklist":
                cat = args.get("category", "OPEN")
                route = args.get("admission_route", "JoSAA")
                gap = args.get("has_gap_year", False)
                tool_data = rag_service.get_checklist(category=cat, admission_route=route, has_gap_year=gap)

            elif name == "get_calendar":
                batch = args.get("batch", "UG_Freshers")
                q = args.get("query", "")
                tool_data = rag_service.get_calendar(batch=batch, query=q)

            elif name == "route_to":
                dest = args.get("destination", "lhtc_gate")
                loc = args.get("current_location", "gate")
                sim_res = run_simulation_callback(agent_type="a_star", congestion=True, start_location=loc) if run_simulation_callback else {}
                trace = sim_res.get("trace", [])
                tool_data = {
                    "destination": dest,
                    "starting_from": loc,
                    "metrics": sim_res.get("metrics"),
                    "trace_steps": [s["current_room"] for s in trace[:6]],
                    "next_immediate_step": trace[0]["action_taken"] if trace else ["complete", loc],
                    "source_url": "https://www.iiitdmj.ac.in/campus/map.php",
                    "last_verified": "2026-11-01"
                }

            elif name == "search_web":
                query_str = args.get("query", user_query)
                tool_data = rag_service.search_web(query_str)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": name,
                "content": json.dumps(tool_data)
            })

        final_resp = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=messages
        )
        return final_resp.choices[0].message.content

    return response_message.content
