import re
from itertools import combinations
from typing import List, Optional, Set, Dict, Tuple
import requests

CONCEPTNET_API = "https://api.conceptnet.io"

DEFAULT_RELATIONS: Set[str] = {
    "IsA", "PartOf", "HasA",
    "UsedFor", "CapableOf", "ReceivesAction",
    "AtLocation", "LocatedNear", "HasProperty",
}

def _norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s

def _to_en_uri(concept: str) -> str:
    return f"/c/en/{_norm(concept)}"

def _as_label(end: dict) -> str:
    if isinstance(end, dict):
        if end.get("label"):
            return str(end["label"]).strip()
        _id = end.get("@id")
        if isinstance(_id, str):
            return _id.split("/")[-1].replace("_", " ").strip()
    return ""

def _weight(edge: dict) -> float:
    w = edge.get("weight", 0.0)
    try:
        return float(w)
    except Exception:
        return 0.0

def _choose_article(phrase: str) -> str:
    """Very simple a/an heuristic."""
    w = (phrase or "").strip().lower()
    if not w:
        return "a"
    # crude vowel sound heuristic
    return "an" if w[0] in "aeiou" else "a"

def _maybe_article(noun: str) -> str:
    """Add a/an unless it already looks like it has a determiner."""
    n = (noun or "").strip()
    if not n:
        return n
    lower = n.lower()
    if lower.startswith(("a ", "an ", "the ", "this ", "that ", "these ", "those ")):
        return n
    # If it's plural-ish, don't force an article (very crude).
    if lower.endswith("s") and not lower.endswith(("ss", "us")):
        return n
    return f"{_choose_article(n)} {n}"

REL_TEMPLATES = {
    "IsA":            lambda s, o: f"{_maybe_article(s)} is { _maybe_article(o) if not o.lower().startswith(('a ','an ','the ')) else o }",
    "PartOf":         lambda s, o: f"{_maybe_article(s)} is part of {_maybe_article(o)}",
    "HasA":           lambda s, o: f"{_maybe_article(s)} has {_maybe_article(o)}",
    "UsedFor":        lambda s, o: f"{_maybe_article(s)} is used for {o}",
    "CapableOf":      lambda s, o: f"{_maybe_article(s)} can {o}",
    "ReceivesAction": lambda s, o: f"{_maybe_article(s)} can be {o}",
    "AtLocation":     lambda s, o: f"{_maybe_article(s)} is often found in {_maybe_article(o)}",
    "LocatedNear":    lambda s, o: f"{_maybe_article(s)} is often near {_maybe_article(o)}",
    "HasProperty":    lambda s, o: f"{_maybe_article(s)} can be {o}",
}

def _edge_to_sentence(start: str, rel: str, end: str) -> str:
    start = start.replace("_", " ").strip()
    end = end.replace("_", " ").strip()
    if rel in REL_TEMPLATES:
        sent = REL_TEMPLATES[rel](start, end)
    else:
        sent = f"{_maybe_article(start)} {rel} {end}"
    # light cleanup
    sent = re.sub(r"\s+", " ", sent).strip()
    # sentence-case + period
    return sent[0].upper() + sent[1:] + "."

def get_conceptnet_facts_for_image(
    objects: List[str],
    *,
    relations: Optional[Set[str]] = None,
    per_object_limit: int = 25,
    pair_limit: int = 25,
    min_weight: float = 1.0,
    max_facts: int = 40,
) -> List[str]:
    """
    Returns up to `max_facts` ConceptNet facts for an image, sorted by edge weight (desc),
    rendered as simple natural-language sentences.

    Facts are collected from:
      - per-object edges
      - direct edges connecting every pair of objects
    """
    rels = relations or DEFAULT_RELATIONS

    # de-dup objects (preserve order)
    seen = set()
    objs = []
    for o in objects:
        o2 = o.strip()
        if o2 and o2.lower() not in seen:
            seen.add(o2.lower())
            objs.append(o2)

    def keep_edge(e: dict) -> bool:
        rel = e.get("rel", {}).get("label")
        return (rel in rels) and (_weight(e) >= min_weight)

    # Deduplicate by (start, rel, end) and keep the max weight we saw.
    best: Dict[Tuple[str, str, str], float] = {}

    def add_edge_fact(e: dict):
        start = _as_label(e.get("start", {}))
        rel = e.get("rel", {}).get("label", "")
        end = _as_label(e.get("end", {}))
        if not (start and rel and end):
            return
        key = (start, rel, end)
        w = _weight(e)
        if w > best.get(key, -1.0):
            best[key] = w

    # 1) per-object edges
    for obj in objs:
        uri = _to_en_uri(obj)
        print(requests.get(f"{CONCEPTNET_API}/query?start={uri}?limit={per_object_limit}"))
        data = requests.get(f"{CONCEPTNET_API}{uri}?limit={per_object_limit}").json()
        
        for e in (data.get("edges") or []):
            if keep_edge(e):
                add_edge_fact(e)

    # 2) direct links between every pair of objects
    for a, b in combinations(objs, 2):
        a_uri, b_uri = _to_en_uri(a), _to_en_uri(b)
        data = requests.get(
            f"{CONCEPTNET_API}/query?node={a_uri}&other={b_uri}&limit={pair_limit}"
        ).json()
        for e in (data.get("edges") or []):
            if keep_edge(e):
                add_edge_fact(e)

    # Sort by weight (desc) and render as sentences
    ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:max_facts]
    sentences = [_edge_to_sentence(s, r, o) for (s, r, o), _w in ranked]
    return sentences
