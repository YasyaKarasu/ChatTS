import json
import re
from difflib import SequenceMatcher


def parse_json_array(response: str):
    try:
        data = json.loads(response)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def normalize_text(text) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def name_match_score(query_name: str, candidate_label: str) -> float:
    q = normalize_text(query_name)
    c = normalize_text(candidate_label)
    if not q or not c:
        return 0.0

    ratio = SequenceMatcher(None, q, c).ratio()
    q_tokens = set(q.split())
    c_tokens = set(c.split())
    overlap = len(q_tokens & c_tokens) / max(len(q_tokens), 1)
    return 0.7 * ratio + 0.3 * overlap
