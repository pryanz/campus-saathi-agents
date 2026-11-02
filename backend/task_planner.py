"""
Intelligent "What should I do first?" Task-Order Planner for arriving students.
Incorporate arrival time, luggage status, and live queue observations to output optimal sequence
and human-readable step-by-step navigation directions.
"""
from datetime import datetime
from envi.campus_graph import CampusGraph

CAMPUS_GRAPH = CampusGraph("data/campus_map.json")

STATION_METADATA = {
    "gate": {
        "title": "Main Campus Security Gate",
        "location": "Institute Entrance Gate",
        "action": "Present JoSAA Provisional Seat Allotment letter to security to collect color-coded reporting token and student handbook.",
        "docs_needed": ["Provisional Seat Allotment Letter", "Government Photo ID (Aadhaar / Voter ID)"],
        "mean_service": 2.0
    },
    "lhtc_gate": {
        "title": "LHTC Screening & Docket Collection",
        "location": "LHTC Main Entrance Foyer",
        "action": "Staff checks basic document readiness, issues official university physical reporting docket file, and confirms counter sequence.",
        "docs_needed": ["Reporting Token", "Allotment Letter Copy"],
        "mean_service": 3.0
    },
    "L-103": {
        "title": "Document Verification Counter",
        "location": "Room L-103, LHTC Ground Floor",
        "action": "Physical verification of Class 10th/12th marksheets, Category Certificate (Central format), Medical Certificate, and Affidavits. Verified files receive green stamp.",
        "docs_needed": ["Original 10th & 12th Marksheets + 2 copies", "Category Certificate (OBC-NCL/EWS/SC/ST)", "Medical Certificate (Annexure 8)", "Gap Affidavit (if dropper)"],
        "mean_service": 5.0
    },
    "L-104": {
        "title": "Fee Clearance & Accounts Reconciliation",
        "location": "Room L-104, LHTC Ground Floor",
        "action": "Accounts officer reconciles JoSAA SAF (Rs 35k/17.5k) and PAF (Rs 45k/20k) receipts. Balance fee paid via POS card / UPI / net banking.",
        "docs_needed": ["SAF & PAF Payment Receipts", "Debit/Credit Card or Net Banking for balance fee"],
        "mean_service": 10.0
    },
    "L-105": {
        "title": "Department & Academic Roll No. Registration",
        "location": "Room L-105, LHTC Ground Floor",
        "action": "Institute Roll Number generated in ERP system (e.g. 2026BCS001). Official batch timetable and course syllabus booklet issued.",
        "docs_needed": ["Green-stamped reporting dossier from L-103", "Fee clearance receipt from L-104"],
        "mean_service": 5.0
    },
    "L-106": {
        "title": "ID Card Biometrics & Photo Capture",
        "location": "Room L-106, LHTC Ground Floor",
        "action": "Live webcam photograph taken and thumb biometrics recorded for permanent Smart RFID Student Identity Card.",
        "docs_needed": ["Roll Number Slip from L-105"],
        "mean_service": 10.0
    },
    "L-206": {
        "title": "Library Membership Activation",
        "location": "Room L-206, LHTC 1st Floor",
        "action": "Central Library digital OPAC access credentials activated and Book Bank scheme registration completed.",
        "docs_needed": ["Roll Number Slip", "Passport size photograph"],
        "mean_service": 2.0
    },
    "panini_gate": {
        "title": "Hostel Allotment & Warden Desk",
        "location": "Panini Gate / Hall 1 & 4 Ground Floor",
        "action": "Present green-stamped dossier or allotment letter to receive hostel room allocation slip and mess registration card.",
        "docs_needed": ["Reporting Dossier / Allotment Slip", "Mess Caution Deposit Receipt"],
        "mean_service": 2.0
    },
    "room": {
        "title": "Hostel Room Handover & Luggage Drop",
        "location": "Allocated Hall of Residence Room",
        "action": "Meet caretaker to collect room keys. Inspect room amenities (fan, bed, study table, wardrobe lock) and safely lock heavy suitcases.",
        "docs_needed": ["Room Allocation Slip from Panini Gate"],
        "mean_service": 10.0
    },
    "office": {
        "title": "Hostel Caretaker & Warden Clearance",
        "location": "Hostel Office, Hall of Residence",
        "action": "Submit physical room handover sheet signed by student and caretaker; register local guardian contact details.",
        "docs_needed": ["Signed Inventory Handover Sheet", "Local Guardian Undertaking"],
        "mean_service": 5.0
    },
    "shop": {
        "title": "Campus Shopping Complex (Hostel Necessities)",
        "location": "Shopping Complex (adjacent to Core Lab)",
        "action": "Purchase mattress, pillow, bedsheet, bucket, mug, extension cord, umbrella, and room lock.",
        "docs_needed": ["Payment mode (UPI / Cash)"],
        "mean_service": 10.0
    }
}


def _parse_hour(time_str: str) -> float:
    # time string se float hour nikalenge (e.g. '09:30 AM' -> 9.5)
    s = time_str.strip().upper()
    try:
        if "AM" in s or "PM" in s:
            dt = datetime.strptime(s, "%I:%M %p") if ":" in s else datetime.strptime(s, "%I %p")
            return dt.hour + (dt.minute / 60.0)
        parts = s.split(":")
        hr = float(parts[0])
        mn = float(parts[1]) if len(parts) > 1 else 0.0
        return hr + (mn / 60.0)
    except Exception:
        return 10.0 # default 10:00 AM


def plan_first_steps(arrival_time: str = "09:30 AM", has_heavy_luggage: bool = True, category: str = "OPEN", current_queues: dict = None) -> dict:
    # arrival time aur bag status ke hisab se optimal sequence plan karenge
    hour = _parse_hour(arrival_time)
    queues = current_queues or {}

    acad_corridor = ["lhtc_gate", "L-103", "L-104", "L-105", "L-106", "L-206"]
    hostel_corridor = ["panini_gate", "room", "office", "shop"]

    rationale_points = []

    # 1. Luggage Decision
    if has_heavy_luggage:
        rationale_points.append(
            "🧳 **Heavy Luggage Strategy:** Carrying heavy suitcases between academic lecture halls adds substantial fatigue (+40% travel friction). "
            "We recommend visiting the Hostel Office (Panini Gate) first to collect your room key and drop your luggage safely in your room before reporting to LHTC."
        )
        hostel_first = True
    else:
        rationale_points.append(
            "🎒 **Backpack Only:** Since you have light luggage, you are free to head directly to the Academic Complex (LHTC) to complete document verification without detours."
        )
        hostel_first = False

    # 2. Arrival Time Decision
    if hour < 10.0:
        time_context = "Early Morning (08:00 - 10:00 AM) — Counters are opening; queues are short."
        if not has_heavy_luggage:
            hostel_first = False
    elif 10.0 <= hour <= 13.5:
        time_context = "Peak Morning Rush (10:00 AM - 01:30 PM) — High queue wait times at Document Verification Counter (L-103)."
        if has_heavy_luggage:
            rationale_points.append(
                "⏰ **Peak Rush Avoidance:** Document Verification (L-103) is at peak queue. Dropping bags at the hostel first allows the morning verification backlog to clear."
            )
            hostel_first = True
    elif 13.5 < hour <= 15.5:
        time_context = "Afternoon Session (01:30 - 03:30 PM) — Post-lunch resumption; moderate steady queues."
    else:
        time_context = "Late Afternoon (After 03:30 PM) — Accounts (L-104) and Warden Office close by 05:00 PM."
        rationale_points.append(
            "⚠️ **Counter Closing Alert:** Fee Clearance (L-104) and Warden Office close around 05:00 PM. Prioritize fee clearance and token stamping immediately!"
        )
        # Even with bags, must get accounts cleared before closing
        hostel_first = False

    # 3. Assemble Sequence
    if hostel_first:
        sequence = ["gate"] + hostel_corridor + acad_corridor
    else:
        sequence = ["gate"] + acad_corridor + hostel_corridor

    # 4. Generate Human-Readable Step-by-Step Directions
    directions = []
    total_travel = 0.0
    total_wait = 0.0
    total_serv = 0.0

    for idx, station in enumerate(sequence):
        meta = STATION_METADATA.get(station, {})
        serv = float(meta.get("mean_service", 3.0))
        wait = float(queues.get(station, 2.0))
        travel = 0.0

        if idx > 0:
            prev_station = sequence[idx - 1]
            travel = float(CAMPUS_GRAPH.shortest_path_weight(prev_station, station))

        total_travel += travel
        total_wait += wait
        total_serv += serv

        directions.append({
            "step_number": idx + 1,
            "station_code": station,
            "title": meta.get("title", station),
            "location": meta.get("location", ""),
            "action": meta.get("action", ""),
            "docs_needed": meta.get("docs_needed", []),
            "estimated_wait_min": round(wait, 1),
            "service_time_min": serv,
            "walking_time_min": travel
        })

    estimated_total = round(total_travel + total_wait + total_serv, 1)

    return {
        "status": "success",
        "arrival_time": arrival_time,
        "time_context": time_context,
        "has_heavy_luggage": has_heavy_luggage,
        "category": category,
        "recommended_first_stop": sequence[1] if len(sequence) > 1 else sequence[0],
        "recommended_first_stop_name": STATION_METADATA.get(sequence[1], {}).get("title", ""),
        "recommended_sequence": sequence,
        "estimated_total_time_min": estimated_total,
        "time_breakdown": {
            "walking_time": round(total_travel, 1),
            "queue_wait_time": round(total_wait, 1),
            "service_time": round(total_serv, 1)
        },
        "rationale": rationale_points,
        "directions": directions
    }
