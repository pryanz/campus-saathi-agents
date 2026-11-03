import copy
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from algorithms.astar import AStarAgent
from algorithms.bfs import BFSAgent
from algorithms.dfs import DFSAgent
from algorithms.greedy_nearest import GreedyNearestAgent
from algorithms.ids import IDSAgent
from algorithms.simulated_annealing import SimulatedAnnealingAgent
from algorithms.ucs import UCSAgent
from algorithms.value_iteration import ValueIterationAgent
from core.problem import CampusProblem
from core.profile import StudentProfile
from core.rules_engine import (RulePackError, build_checklist, load_rule_pack,
                               matches, validate_rule_pack)
from envi.campus_graph import CampusGraph
from envi.simulator import CampusSimulator

MAP = os.path.join(ROOT, "data", "campus_map.json")
STEPS = os.path.join(ROOT, "data", "steps.json")
HOSTEL_STATIONS = {"panini_gate", "room", "office"}


def doc_ids(checklist):
    return {d.id for d in checklist.documents}


def _steps():
    with open(STEPS) as f:
        return json.load(f)


# ------------------------------------------------- six+ distinct profiles

def test_open_student_gets_only_the_common_documents():
    c = build_checklist(StudentProfile(category="OPEN", home_state="Madhya Pradesh"))
    ids = doc_ids(c)
    assert {"allotment_letter", "class10_marksheet", "class12_marksheet",
            "hostel_allotment_letter"} <= ids
    for reservation_doc in ("obc_ncl_certificate", "ews_certificate",
                            "sc_caste_certificate", "st_caste_certificate",
                            "pwd_certificate", "domicile_certificate"):
        assert reservation_doc not in ids


def test_obc_ncl_and_open_receive_distinct_checklists():
    open_c = build_checklist(StudentProfile("OPEN", "Madhya Pradesh"))
    obc_c = build_checklist(StudentProfile("OBC-NCL", "Madhya Pradesh"))
    assert doc_ids(obc_c) - doc_ids(open_c) == {"obc_ncl_certificate"}
    assert doc_ids(open_c) < doc_ids(obc_c)
    cert = next(d for d in obc_c.documents if d.id == "obc_ncl_certificate")
    assert cert.original_copies == 1 and cert.photocopies == 2
    assert cert.issued_on_or_after == "2026-04-01"


def test_ews_student_needs_ews_certificate_only():
    ids = doc_ids(build_checklist(StudentProfile("EWS", "Madhya Pradesh")))
    assert "ews_certificate" in ids
    assert not {"obc_ncl_certificate", "sc_caste_certificate"} & ids


def test_sc_out_of_state_needs_domicile_but_in_state_does_not():
    out = doc_ids(build_checklist(StudentProfile("SC", "Bihar")))
    inn = doc_ids(build_checklist(StudentProfile("SC", "Madhya Pradesh")))
    assert {"sc_caste_certificate", "domicile_certificate"} <= out
    assert "sc_caste_certificate" in inn
    assert "domicile_certificate" not in inn


def test_st_student_needs_st_certificate():
    ids = doc_ids(build_checklist(StudentProfile("ST", "Madhya Pradesh")))
    assert "st_caste_certificate" in ids
    assert "sc_caste_certificate" not in ids


def test_pwd_flag_adds_disability_certificate_regardless_of_category():
    open_pwd = doc_ids(build_checklist(StudentProfile("OPEN", "Kerala", pwd=True)))
    obc_pwd = doc_ids(build_checklist(StudentProfile("OBC-NCL", "Kerala", pwd=True)))
    assert "pwd_certificate" in open_pwd and "pwd_certificate" in obc_pwd
    assert "pwd_certificate" not in doc_ids(build_checklist(StudentProfile("OPEN", "Kerala")))


def test_admission_route_changes_documents():
    josaa = doc_ids(build_checklist(StudentProfile("OPEN", "Kerala", "JoSAA")))
    csab = doc_ids(build_checklist(StudentProfile("OPEN", "Kerala", "CSAB")))
    dasa = doc_ids(build_checklist(StudentProfile("OPEN", "Kerala", "DASA")))
    assert "migration_certificate" in csab - josaa
    assert "dasa_documents" in dasa - josaa
    assert "dasa_documents" not in csab


def test_day_scholar_skips_hostel_stations_and_documents():
    c = build_checklist(StudentProfile("OPEN", "Madhya Pradesh", hostel_required=False))
    assert not HOSTEL_STATIONS & set(c.required_stations)
    assert "hostel_allotment_letter" not in doc_ids(c)
    assert "shop" in c.required_stations

    with_hostel = build_checklist(StudentProfile("OPEN", "Madhya Pradesh"))
    assert HOSTEL_STATIONS <= set(with_hostel.required_stations)


def test_matching_is_case_and_whitespace_insensitive():
    a = build_checklist(StudentProfile("obc-ncl ", " bihar", " csab "))
    b = build_checklist(StudentProfile("OBC-NCL", "Bihar", "CSAB"))
    assert doc_ids(a) == doc_ids(b)


def test_documents_are_grouped_under_required_stations():
    c = build_checklist(StudentProfile("OBC-NCL", "Bihar"))
    grouped = c.documents_by_station()
    assert set(grouped) >= {d.station for d in c.documents}
    assert all(d.station == st for st, docs in grouped.items() for d in docs)
    assert {d.station for d in c.documents} <= set(c.required_stations)


def test_checklist_serialises_to_json():
    c = build_checklist(StudentProfile("OBC-NCL", "Bihar"))
    data = json.loads(json.dumps(c.to_dict()))
    assert data["rule_pack_version"] == "v1"
    assert data["profile"]["category"] == "OBC-NCL"
    assert any(d["id"] == "obc_ncl_certificate" for d in data["documents"])


# ------------------------------------------------------- data-driven rules

def _write_pack(tmp_path, pack, version="v9"):
    pack = copy.deepcopy(pack)
    pack["version"] = version
    (tmp_path / f"rule_pack_{version}.json").write_text(json.dumps(pack))
    return str(tmp_path)


def test_editing_json_changes_outcome_without_code_changes(tmp_path):
    pack = load_rule_pack("v1")
    for d in pack["documents"]:
        if d["id"] == "obc_ncl_certificate":
            d["copies"] = {"original": 1, "photocopy": 5}
            d["cutoff_date"] = "2026-12-31"
    rules_dir = _write_pack(tmp_path, pack)
    c = build_checklist(StudentProfile("OBC-NCL", "Bihar"), "v9", rules_dir=rules_dir)
    cert = next(d for d in c.documents if d.id == "obc_ncl_certificate")
    assert cert.photocopies == 5
    assert cert.cutoff_date == "2026-12-31"
    assert c.rule_pack_version == "v9"


def test_rule_packs_are_versioned_independently(tmp_path):
    pack = load_rule_pack("v1")
    pack["documents"] = [d for d in pack["documents"] if d["id"] != "obc_ncl_certificate"]
    rules_dir = _write_pack(tmp_path, pack, version="v2")
    profile = StudentProfile("OBC-NCL", "Bihar")
    assert "obc_ncl_certificate" in doc_ids(build_checklist(profile, "v1"))
    assert "obc_ncl_certificate" not in doc_ids(build_checklist(profile, "v2", rules_dir=rules_dir))


def test_shipped_rule_pack_is_valid_and_schema_is_parseable():
    validate_rule_pack(load_rule_pack("v1"))
    with open(os.path.join(ROOT, "data", "rules", "rule_pack.schema.json")) as f:
        schema = json.load(f)
    assert set(schema["required"]) == {"version", "stations", "documents"}


def test_every_station_in_the_pack_exists_in_steps_json():
    pack_stations = {s["id"] for s in load_rule_pack("v1")["stations"]}
    assert pack_stations == set(_steps())


def test_no_category_names_are_hardcoded_in_python_logic():
    with open(os.path.join(ROOT, "core", "rules_engine.py")) as f:
        source = f.read()
    for literal in ("OBC", "EWS", "OPEN", "JoSAA", "CSAB", "Madhya"):
        assert literal not in source


@pytest.mark.parametrize("mutate, message", [
    (lambda p: p["documents"][0].update(station="nowhere"), "unknown station"),
    (lambda p: p["documents"][0].update(when={"shoe_size": ["9"]}), "unknown profile field"),
    (lambda p: p["documents"].append(copy.deepcopy(p["documents"][0])), "Duplicate document"),
    (lambda p: p["documents"][0].update(cutoff_date="31-12-2026"), "invalid cutoff_date"),
    (lambda p: p["documents"][0].update(copies={"original": -1, "photocopy": 1}), "non-negative"),
    (lambda p: p.pop("stations"), "missing required key"),
    (lambda p: p.update(surprise=1), "Unknown top-level"),
])
def test_invalid_rule_packs_are_rejected(mutate, message):
    pack = load_rule_pack("v1")
    mutate(pack)
    with pytest.raises(RulePackError, match=message):
        validate_rule_pack(pack)


def test_missing_and_malformed_versions_raise():
    with pytest.raises(RulePackError, match="not found"):
        build_checklist(StudentProfile(), "v404")
    with pytest.raises(RulePackError, match="Invalid rule pack version"):
        build_checklist(StudentProfile(), "../etc/passwd")


def test_matches_helper_semantics():
    p = StudentProfile("SC", "Bihar")
    assert matches(None, p) and matches({}, p)
    assert matches({"category": ["SC", "ST"]}, p)
    assert not matches({"category": ["OPEN"]}, p)
    assert matches({"home_state": {"not": ["Madhya Pradesh"]}}, p)
    assert not matches({"category": ["SC"], "home_state": {"not": ["Bihar"]}}, p)


def test_profile_validation():
    with pytest.raises(ValueError):
        StudentProfile(category="  ")
    with pytest.raises(ValueError, match="Unknown profile fields"):
        StudentProfile.from_dict({"category": "OPEN", "height": 180})
    assert StudentProfile.from_dict({"category": "ST"}).category == "ST"


# --------------------------------------------- CampusProblem goal_steps

def test_problem_without_profile_is_unchanged():
    graph = CampusGraph(MAP)
    steps = _steps()
    problem = CampusProblem(graph, steps)
    assert problem.goal_steps == set(steps)
    assert problem.steps_config is steps
    assert problem.checklist is None


def test_goal_steps_follow_the_profile():
    graph = CampusGraph(MAP)
    steps = _steps()
    hostel = CampusProblem(graph, steps, profile=StudentProfile("OPEN", "Bihar"))
    day = CampusProblem(graph, steps, profile=StudentProfile("OPEN", "Bihar", hostel_required=False))
    assert hostel.goal_steps == set(steps)
    assert day.goal_steps == set(steps) - HOSTEL_STATIONS
    assert set(day.steps_config) == day.goal_steps
    assert day.checklist.required_stations == [s for s in steps if s in day.goal_steps]
    assert not day.goal_test(day.initial_state())
    done = ("gate", frozenset(day.goal_steps))
    assert day.goal_test(done)


def test_explicit_required_stations_and_unknown_station_error():
    graph = CampusGraph(MAP)
    problem = CampusProblem(graph, _steps(), required_stations=["gate", "shop"])
    assert problem.goal_steps == {"gate", "shop"}
    with pytest.raises(ValueError, match="not in steps config"):
        CampusProblem(graph, _steps(), required_stations=["gate", "moon_base"])


# ------------------- search agents still run on a profile-driven problem

AGENTS = [BFSAgent, DFSAgent, GreedyNearestAgent, AStarAgent, UCSAgent,
          ValueIterationAgent, SimulatedAnnealingAgent, IDSAgent]


@pytest.mark.parametrize("agent_cls", AGENTS, ids=lambda c: c.__name__)
def test_agents_complete_exactly_the_required_stations(agent_cls):
    profile = StudentProfile("OBC-NCL", "Bihar", hostel_required=False)
    simulator = CampusSimulator(MAP, STEPS, seed=42, profile=profile)
    required = simulator.problem.goal_steps
    assert required and not HOSTEL_STATIONS & required

    summary = simulator.run_simulation(agent_cls(simulator.problem), max_steps=500,
                                       congestion=False)
    completed = set(summary["trace"][-1]["completed_steps"])
    assert completed == required
    assert simulator.problem.goal_test(("gate", frozenset(completed)))


def test_simulator_default_still_visits_all_eleven_stations():
    simulator = CampusSimulator(MAP, STEPS, seed=42)
    summary = simulator.run_simulation(AStarAgent(simulator.problem), max_steps=500,
                                       congestion=False)
    assert len(summary["trace"][-1]["completed_steps"]) == 11
