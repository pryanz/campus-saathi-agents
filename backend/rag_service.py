import json
import re
import math
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ALLOWLISTED_DOMAINS = ["iiitdmj.ac.in", "josaa.nic.in", "antiragging.in"]

# common filler words jo matching me ignore hone chahiye
STOP_WORDS = {
    "the", "is", "at", "in", "on", "of", "a", "an", "to", "for", "and", "or",
    "it", "do", "my", "me", "we", "if", "be", "by", "so", "no", "not", "can",
    "how", "what", "when", "where", "which", "who", "will", "are", "was", "has",
    "have", "had", "this", "that", "with", "from", "about", "i", "am", "there",
    "tell", "please", "should", "would", "could", "does", "did", "get", "got",
    "campus", "institute", "college", "iiitdm", "jabalpur", "student", "students",
    "office", "hours", "department", "apply", "know", "need",
}

class CampusRAGService:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.chunks = []
        self.documents_kb = []
        self.places_kb = []
        self.calendar_kb = {}
        self.kb_faq = []
        self._load_datasets() # saara official data load karenge

    def _tokenize(self, text: str) -> list[str]:
        cleaned = re.sub(r'[^a-zA-Z0-9_\-\s]', ' ', text.lower())
        return [t.strip() for t in cleaned.split() if len(t.strip()) > 1]

    def _load_datasets(self):
        # documents data load aur chunking
        doc_path = self.data_dir / "documents.json"
        if doc_path.exists():
            with open(doc_path, "r", encoding="utf-8") as f:
                self.documents_kb = json.load(f)
            for doc in self.documents_kb:
                text = (
                    f"Document: {doc['name']}. Purpose: {doc.get('purpose', '')}. "
                    f"Applicability: {', '.join(doc.get('applicability', []))}. "
                    f"Format: {doc.get('format_required', '')}. "
                    f"Cutoff Rules: {doc.get('validity_cutoff', '')}. "
                    f"Action If Missing / Issue: {doc.get('action_if_missing', '')}"
                )
                self.chunks.append({
                    "type": "document",
                    "id": doc["id"],
                    "title": doc["name"],
                    "text": text,
                    "source_url": doc.get("source_url", "https://josaa.nic.in"),
                    "last_verified": doc.get("last_verified", "2026-10-15"),
                    "tokens": Counter(self._tokenize(text + " " + doc["name"] * 3)),
                    "raw": doc
                })

        # campus places aur facilities data
        places_path = self.data_dir / "places.json"
        if places_path.exists():
            with open(places_path, "r", encoding="utf-8") as f:
                self.places_kb = json.load(f)
            for pl in self.places_kb:
                services = ", ".join(pl.get("services", []))
                text = (
                    f"Place/Facility: {pl['name']}. Location: {pl.get('location', '')}. "
                    f"Operating Hours: {pl.get('operating_hours', '')}. Services: {services}. "
                    f"Rates: {pl.get('rates', 'Standard')}."
                )
                self.chunks.append({
                    "type": "place",
                    "id": pl["id"],
                    "title": pl["name"],
                    "text": text,
                    "source_url": pl.get("source_url", "https://www.iiitdmj.ac.in"),
                    "last_verified": pl.get("last_verified", "2026-10-25"),
                    "tokens": Counter(self._tokenize(text + " " + pl["name"] * 3)),
                    "raw": pl
                })

        # academic calendar aur reporting dates
        cal_path = self.data_dir / "calendar.json"
        if cal_path.exists():
            with open(cal_path, "r", encoding="utf-8") as f:
                self.calendar_kb = json.load(f)
            cal_url = self.calendar_kb.get("source_url", "https://www.iiitdmj.ac.in/academics/calendar.pdf")
            cal_ver = self.calendar_kb.get("last_verified", "2026-10-25")
            for ev in self.calendar_kb.get("events", []):
                text = (
                    f"Academic Event: {ev['title']} on Date: {ev['date']} Time: {ev.get('time', '')}. "
                    f"Venue: {ev.get('venue', '')}. Details: {ev.get('description', '')}"
                )
                self.chunks.append({
                    "type": "calendar",
                    "id": f"event_{ev['date']}",
                    "title": ev["title"],
                    "text": text,
                    "source_url": cal_url,
                    "last_verified": cal_ver,
                    "tokens": Counter(self._tokenize(text + " " + ev["title"] * 2)),
                    "raw": ev
                })

        # common reporting FAQs
        kb_path = self.data_dir / "campus_kb.json"
        if kb_path.exists():
            with open(kb_path, "r", encoding="utf-8") as f:
                kb_data = json.load(f)
            for idx, faq in enumerate(kb_data.get("faqs", [])):
                text = f"FAQ: {faq['question']} Answer: {faq['answer']}"
                self.chunks.append({
                    "type": "faq",
                    "id": f"faq_{idx}",
                    "title": faq["question"],
                    "text": text,
                    "source_url": "https://www.iiitdmj.ac.in/admissions/faq_2026.pdf",
                    "last_verified": "2026-10-30",
                    "tokens": Counter(self._tokenize(text + " " + faq["question"] * 3)),
                    "raw": faq
                })

    def search_vector(self, query: str, top_k: int = 3, type_filter: str = None) -> list[dict]:
        # query tokens match karke relevant chunks nikalenge
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        q_counts = Counter(query_tokens)
        scores = []

        for chunk in self.chunks:
            if type_filter and chunk["type"] != type_filter:
                continue

            score = 0.0
            chunk_tokens = chunk["tokens"]
            chunk_len = sum(chunk_tokens.values())

            for token, q_weight in q_counts.items():
                if token in chunk_tokens:
                    tf = chunk_tokens[token]
                    tf_score = 1.0 + math.log(tf)
                    norm_score = tf_score / (1.0 + 0.001 * chunk_len)
                    score += q_weight * norm_score * 2.5

            # exact title match hone par boost
            lowered_title = chunk["title"].lower()
            if any(term in lowered_title for term in query_tokens):
                score += 3.5
            if query.lower() in lowered_title:
                score += 6.0

            if score > 0:
                scores.append((score, chunk))

        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for s, chunk in scores[:top_k]:
            results.append({
                "type": chunk["type"],
                "title": chunk["title"],
                "content": chunk["text"],
                "source_url": chunk["source_url"],
                "last_verified": chunk["last_verified"],
                "score": round(s, 2),
                "raw": chunk["raw"]
            })
        return results

    def get_facility(self, keyword_or_name: str) -> dict:
        # xerox, notary, doctor waghera keyword se facility dhundenge
        key = keyword_or_name.lower().strip()
        for pl in self.places_kb:
            if key in pl["name"].lower() or key in pl.get("location", "").lower():
                return pl
            for s in pl.get("services", []):
                if key in s.lower():
                    return pl

        res = self.search_vector(keyword_or_name, top_k=1, type_filter="place")
        return res[0]["raw"] if res else None

    def get_checklist(self, category: str = "OPEN", admission_route: str = "JoSAA", has_gap_year: bool = False) -> dict:
        # category aur gap year ke hisab se filtered documents list
        cat_clean = category.upper().strip()
        matched_docs = []

        for doc in self.documents_kb:
            app = doc.get("applicability", [])
            is_applicable = (
                "OPEN" in app or
                cat_clean in app or
                (cat_clean in ["OBC", "OBC-NCL"] and "OBC-NCL" in app) or
                (cat_clean in ["EWS", "GEN-EWS"] and "GEN-EWS" in app) or
                (cat_clean in ["SC", "ST"] and cat_clean in app) or
                (has_gap_year and ("DROPPERS" in app or "YEAR_GAP" in app))
            )

            if is_applicable:
                matched_docs.append({
                    "document_name": doc["name"],
                    "format_required": doc.get("format_required"),
                    "validity_cutoff": doc.get("validity_cutoff"),
                    "action_if_missing": doc.get("action_if_missing"),
                    "source_url": doc.get("source_url"),
                    "last_verified": doc.get("last_verified")
                })

        return {
            "category": cat_clean,
            "admission_route": admission_route,
            "total_documents": len(matched_docs),
            "documents": matched_docs,
            "source_url": "https://josaa.nic.in/document-verification-rules-2026.pdf",
            "last_verified": "2026-10-20"
        }

    def get_calendar(self, batch: str = "UG_Freshers", query: str = "") -> dict:
        # reporting dates aur schedule filter
        events = self.calendar_kb.get("events", [])
        if query:
            q_lower = query.lower()
            filtered = [e for e in events if q_lower in e["title"].lower() or q_lower in e.get("description", "").lower()]
            if filtered:
                events = filtered

        return {
            "academic_session": self.calendar_kb.get("academic_session", "2026-2027"),
            "batch": batch,
            "events": events,
            "source_url": self.calendar_kb.get("source_url", "https://www.iiitdmj.ac.in/academics/calendar.pdf"),
            "last_verified": self.calendar_kb.get("last_verified", "2026-10-25")
        }

    def search_web(self, query: str) -> dict:
        # sirf allowlisted domains me hi search restrict karenge
        local_matches = self.search_vector(query, top_k=3)
        filtered_matches = [
            m for m in local_matches
            if any(domain in m.get("source_url", "") for domain in ALLOWLISTED_DOMAINS)
        ]

        # false positive filter: query ke main keywords result me hone chahiye
        query_keywords = {
            t for t in self._tokenize(query) if t not in STOP_WORDS and len(t) > 2
        }
        if query_keywords and filtered_matches:
            verified_matches = []
            for m in filtered_matches:
                result_text = (m.get("content", "") + " " + m.get("title", "")).lower()
                # exact match + partial stem match (pehle 4 chars)
                overlap = sum(
                    1 for kw in query_keywords
                    if kw in result_text or any(kw[:4] in word for word in result_text.split())
                )
                ratio = overlap / len(query_keywords) if query_keywords else 0
                score = m.get("score", 0)
                # high overlap wale results ko thoda relaxed score threshold
                min_score = 2.0 if ratio >= 0.7 else 6.0
                if ratio >= 0.35 and score >= min_score:
                    verified_matches.append(m)
            filtered_matches = verified_matches

        if filtered_matches:
            return {
                "query": query,
                "allowlisted_domains": ALLOWLISTED_DOMAINS,
                "status": "success",
                "results": [
                    {
                        "title": m["title"],
                        "content": m["content"],
                        "source_url": m["source_url"],
                        "last_verified": m["last_verified"]
                    }
                    for m in filtered_matches
                ]
            }

        return {
            "query": query,
            "allowlisted_domains": ALLOWLISTED_DOMAINS,
            "status": "not_found",
            "message": "No verified policy found in official domain sources (iiitdmj.ac.in, josaa.nic.in, antiragging.in).",
            "citation": "Official Institute Helpdesk / Exception Desk L-103",
            "last_verified": "2026-11-01"
        }

rag_service = CampusRAGService()
