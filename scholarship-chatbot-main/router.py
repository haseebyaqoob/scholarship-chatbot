"""
router.py
─────────
2-stage router for the scholarship assistant.

Architecture change vs original (single-pass router):
  OLD: One large prompt asked the model to simultaneously classify intent,
       extract structured CSV filters, extract profile fields, resolve pronouns,
       and enforce schema correctness — ~6 cognitive tasks in one pass.
       Result: 15–30% malformed or partially-correct JSON on a 7B model.

  NEW: Stage 1 — classify intent only (small prompt, ~150 tokens output)
       Stage 2 — extract schema ONLY when needed:
                   CSV route    → _run_csv_extractor()
                   PROFILE route → _run_profile_extractor()
       Result: each call is a single focused task; reliability improves
               significantly because the model never juggles 6 things at once.

Additionally:
  - use_json_format=True on all LLM calls forces Ollama to produce
    syntactically valid JSON at the inference level, not just via instruction.
  - Each stage has its own focused system prompt (much smaller than the
    original combined prompt, reducing middle-prompt dilution).
  - validate_router_output() is unchanged — it still derives all valid field
    sets from the live dataframe at runtime with zero hardcoded column names.
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

import pandas as pd
from rapidfuzz import fuzz, process as rfprocess

from config_loader import cfg
from llm_handler import LocalLLM

_FUZZY_THRESHOLD = int(cfg["fuzzy_match_threshold"])
_ROUTER_TOKENS   = int(cfg["router_max_tokens"])

# ── Route constants ────────────────────────────────────────────────────────────
VALID_ROUTES = {
    "CSV_ONLY", "TXT_ONLY", "RAG",
    "PROFILE_UPDATE", "GREETING", "FAREWELL", "OFF_TOPIC",
}

VALID_OPERATIONS = {
    "rows", "count", "exists", "distinct",
    "aggregate", "compare", "group_by",
}

# Stage 1 intent labels → pipeline route names
_INTENT_TO_ROUTE: dict[str, str] = {
    "CSV":       "CSV_ONLY",
    "TXT_ONLY":  "TXT_ONLY",
    "RAG":       "RAG",
    "PROFILE":   "PROFILE_UPDATE",
    "GREETING":  "GREETING",
    "FAREWELL":  "FAREWELL",
    "OFF_TOPIC": "OFF_TOPIC",
}
VALID_INTENTS = set(_INTENT_TO_ROUTE.keys())


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Intent Classifier
# ══════════════════════════════════════════════════════════════════════════════

def _build_classifier_prompt(
    df: pd.DataFrame,
    name_list: list[str],
    last_scholarships: list[str] | None = None,
) -> str:
    """
    Small, focused prompt for intent classification only.
    Injects catalog and recently discussed scholarships for pronoun resolution.
    Does NOT ask the model to extract filters, profile fields, or nested schemas.
    """
    today   = date.today().strftime("%d %B %Y")
    catalog = "\n".join(f"  - {n}" for n in name_list)
    recent  = (
        "\n".join(f"  - {n}" for n in last_scholarships)
        if last_scholarships else "  (none)"
    )

    return f"""You are an intent classifier for a scholarship assistant. Today: {today}

SCHOLARSHIP CATALOG:
{catalog}

RECENTLY DISCUSSED:
{recent}
Use RECENTLY DISCUSSED to resolve vague references ("that one", "tell me more",
"it", "the first one", "what about that").

Classify the user query into EXACTLY ONE intent:

CSV      — counting, aggregation, sorting, or filtering STRUCTURED FACTS:
           "how many", "which has the lowest GPA", "list merit scholarships",
           "sort by deadline", "count need-based scholarships"

TXT_ONLY — user asks about a SPECIFIC NAMED scholarship in descriptive detail:
           benefits, eligibility narrative, required documents, application
           process, bond requirements, contact info, how to apply

RAG      — discovery, recommendation, open-ended exploration, or comparisons
           that need narrative detail:
           "which suits me", "scholarships for Sindh students",
           "what are my options", "compare Fulbright and DAAD"

PROFILE  — user shares or corrects personal information:
           name, GPA, level of study, field, domicile, nationality

GREETING — ONLY simple standalone greetings with nothing else:
           "hi", "hello", "hey", "good morning", "assalam o alaikum"
           (a greeting + a question → route to the question's intent)

FAREWELL — goodbye expressions:
           "bye", "thanks", "thank you", "take care", "see you", "good night"

OFF_TOPIC— everything else:
           bot meta-questions ("are you an AI", "what can you do", "who made you"),
           sports, politics, recipes, celebrities, general knowledge not tied
           to a specific scholarship

Decision rules (apply in order):
1. "are you an AI" / "what can you do" / "who built you" → OFF_TOPIC, never GREETING
2. vague references → resolve via RECENTLY DISCUSSED → TXT_ONLY or RAG
3. profile info + scholarship question in one message → PROFILE
   (the pipeline will also answer the scholarship question automatically)
4. when in doubt between CSV and RAG → RAG

Output ONLY this JSON object, nothing else:
{{
  "intent": "CSV|TXT_ONLY|RAG|PROFILE|GREETING|FAREWELL|OFF_TOPIC",
  "scholarships": [],
  "reason": "one sentence"
}}

scholarships rules:
- only use names that appear EXACTLY in the SCHOLARSHIP CATALOG above
- never invent, abbreviate, or guess names
- for vague references, populate from RECENTLY DISCUSSED
- for broad/general queries ("all scholarships", "any"), leave []
"""


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2a — CSV Spec Extractor
# ══════════════════════════════════════════════════════════════════════════════

def _build_csv_extractor_prompt(df: pd.DataFrame) -> str:
    """
    Focused prompt for CSV query spec extraction only.
    Only called when Stage 1 classified intent as CSV.
    Prompt is much smaller than the original combined router prompt.
    """
    today         = date.today().strftime("%d %B %Y")
    numeric_cols  = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    text_cols     = [
        c for c in df.columns
        if c not in numeric_cols and c not in datetime_cols
    ]

    return f"""Extract a structured CSV query specification from the user query. Today: {today}

Available columns:
  Text    : {', '.join(text_cols) or 'none'}
  Numeric : {', '.join(numeric_cols) or 'none'}
  Date    : {', '.join(datetime_cols) or 'none'}

Choose ONE operation:
  rows      — list or filter rows (default)
  count     — "how many scholarships..."
  exists    — "is there any scholarship that..."
  distinct  — unique values of one column ("what levels are available")
  aggregate — min/max/avg/sum of a numeric or date column
  compare   — side-by-side comparison of named scholarships on specific fields
  group_by  — group rows by a text column and count

Numeric range syntax — ALWAYS use range operators for numbers, never plain equality:
  "GPA below 3.0"         → {{"gpa_min": {{"lte": 3.0}}}}
  "GPA at least 3.5"      → {{"gpa_min": {{"gte": 3.5}}}}
  "amount over 50000"     → {{"amount_pkr": {{"gt": 50000}}}}
  "need-based, GPA ≤ 3.5" → {{"funding_type": "need-based", "gpa_min": {{"lte": 3.5}}}}

CRITICAL — two different keys, do not confuse them:
  field  (singular, null by default) — ONLY for the 'distinct' operation
  fields (plural, [] by default)     — column selection list for 'rows'/'compare'

Output ONLY this JSON object, nothing else:
{{
  "operation": "rows",
  "filters": {{}},
  "sort": [],
  "limit": null,
  "fields": [],
  "field": null,
  "aggregate": {{"type": null, "field": null}},
  "compare_names": [],
  "compare_fields": [],
  "group_field": null
}}
"""


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2b — Profile Field Extractor
# ══════════════════════════════════════════════════════════════════════════════

def _build_profile_extractor_prompt() -> str:
    """
    Focused prompt for profile field extraction only.
    Only called when Stage 1 classified intent as PROFILE.

    Key design: correction patterns are shown with explicit examples so the
    model extracts the NEW value (Y) from "not X it's Y" patterns, not the
    old value. This is the main failure mode for name corrections on 7B models.
    """
    return """Extract personal profile fields from the user message.

Valid output keys: name, level, field, gpa, domicile, nationality

CORRECTION PATTERNS — extract ONLY the new (correct) value, never the old:
  "not Haseeb, it's Ali"          → {"name": "Ali"}
  "actually it's Ali"             → {"name": "Ali"}
  "I meant Ali, not Haseeb"       → {"name": "Ali"}
  "no, call me Ali"               → {"name": "Ali"}
  "wait, my name is Ali"          → {"name": "Ali"}
  "correction: my GPA is 3.7"    → {"gpa": 3.7}
  "wrong, I study medicine"       → {"field": "Medicine"}
  "that's wrong, I'm from Sindh"  → {"domicile": "Sindh"}

STANDARD PATTERNS:
  "my name is Ali"                             → {"name": "Ali"}
  "my GPA is 3.5"                              → {"gpa": 3.5}
  "I study Computer Science"                   → {"field": "Computer Science"}
  "I'm a postgraduate student"                 → {"level": "postgraduate"}
  "I'm from Punjab"                            → {"domicile": "Punjab"}
  "I'm Pakistani"                              → {"nationality": "Pakistani"}
  "I study CS with GPA 3.5, I'm from Punjab"  → {"field": "Computer Science", "gpa": 3.5, "domicile": "Punjab"}

Rules:
  - gpa MUST be a number (float), NOT a string: "3.5" → 3.5
  - level: preserve the student's exact phrasing (e.g., "undergraduate", "postgrad", "PhD")
  - only extract fields the user explicitly mentions
  - for corrections, extract ONLY the corrected/new value
  - if nothing can be extracted, output {}

Output ONLY valid JSON with extracted fields, nothing else.
"""


# ══════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ══════════════════════════════════════════════════════════════════════════════

def _extract_json(raw: str) -> dict | None:
    """Try several strategies to extract a JSON object from raw LLM output."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$",           "", cleaned)

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if m:
        fragment = m.group(0)
        try:
            return json.loads(fragment)
        except Exception:
            pass
        fragment = re.sub(r",\s*([}\]])", r"\1", fragment)
        fragment = re.sub(r"\bTrue\b",  "true",  fragment)
        fragment = re.sub(r"\bFalse\b", "false", fragment)
        fragment = re.sub(r"\bNone\b",  "null",  fragment)
        try:
            return json.loads(fragment)
        except Exception:
            pass

    return None


# ══════════════════════════════════════════════════════════════════════════════
# Validator — unchanged from original
# ══════════════════════════════════════════════════════════════════════════════

def validate_router_output(
    plan: dict[str, Any],
    df: pd.DataFrame,
    name_list: list[str],
) -> dict[str, Any]:
    """
    Validate and sanitise a raw router plan.
    All valid field sets are derived from the actual dataframe at runtime.
    Raises ValueError with a descriptive message on invalid input.
    """
    numeric_cols   = {c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    datetime_cols  = {c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])}
    all_cols       = set(df.columns)
    sortable_cols  = all_cols
    groupable_cols = all_cols - numeric_cols - datetime_cols
    agg_cols       = numeric_cols | datetime_cols

    # ── Route ─────────────────────────────────────────────────────────────────
    route = plan.get("route")
    if route not in VALID_ROUTES:
        raise ValueError(f"Invalid route '{route}'. Must be one of: {VALID_ROUTES}")

    # ── Scholarship names — fuzzy-validate against the loaded catalog ──────────
    raw_names = plan.get("scholarships") or []
    if not isinstance(raw_names, list):
        raise ValueError("'scholarships' must be a list.")
    resolved_names: list[str] = []
    for candidate in raw_names:
        result = rfprocess.extractOne(
            candidate, name_list, scorer=fuzz.token_set_ratio
        )
        if result and result[1] >= _FUZZY_THRESHOLD:
            resolved_names.append(result[0])
    plan["scholarships"] = list(dict.fromkeys(resolved_names))

    # ── Reason ────────────────────────────────────────────────────────────────
    if not isinstance(plan.get("reason"), str):
        plan["reason"] = ""

    # ── Profile update ────────────────────────────────────────────────────────
    profile_update = plan.get("profile_update") or {}
    if not isinstance(profile_update, dict):
        profile_update = {}
    valid_profile_keys = {"name", "level", "field", "gpa", "domicile", "nationality"}
    plan["profile_update"] = {
        k: v for k, v in profile_update.items()
        if k in valid_profile_keys and v not in (None, "", "null")
    }

    # ── CSV query spec ─────────────────────────────────────────────────────────
    csv_spec = plan.get("csv_query_spec") or {}
    if not isinstance(csv_spec, dict):
        raise ValueError("'csv_query_spec' must be a dict.")

    default_spec: dict[str, Any] = {
        "operation":      "rows",
        "filters":        {},
        "sort":           [],
        "limit":          None,
        "fields":         [],
        "field":          None,
        "aggregate":      {"type": None, "field": None},
        "compare_names":  [],
        "compare_fields": [],
        "group_field":    None,
    }
    merged = {**default_spec, **csv_spec}

    op = merged.get("operation", "rows")
    if op not in VALID_OPERATIONS:
        merged["operation"] = "rows"

    valid_sorts: list[dict] = []
    for s in (merged.get("sort") or []):
        if isinstance(s, dict) and s.get("field") in sortable_cols:
            direction = s.get("direction", "asc")
            if direction not in ("asc", "desc"):
                direction = "asc"
            valid_sorts.append({"field": s["field"], "direction": direction})
    merged["sort"] = valid_sorts

    merged["fields"] = [f for f in (merged.get("fields") or []) if f in all_cols]

    agg = merged.get("aggregate") or {}
    if not isinstance(agg, dict):
        agg = {}
    agg_field = agg.get("field")
    agg_type  = agg.get("type")
    if agg_field and agg_field not in agg_cols:
        agg_field = None
    if agg_type not in ("min", "max", "avg", "sum", None):
        agg_type = None
    merged["aggregate"] = {"type": agg_type, "field": agg_field}

    if merged.get("field") and merged["field"] not in all_cols:
        merged["field"] = None

    if merged.get("group_field") and merged["group_field"] not in groupable_cols:
        merged["group_field"] = None

    merged["compare_fields"] = [
        f for f in (merged.get("compare_fields") or []) if f in all_cols
    ]

    raw_compare = merged.get("compare_names") or []
    resolved_compare: list[str] = []
    for candidate in raw_compare:
        result = rfprocess.extractOne(
            candidate, name_list, scorer=fuzz.token_set_ratio
        )
        if result and result[1] >= _FUZZY_THRESHOLD:
            resolved_compare.append(result[0])
    merged["compare_names"] = list(dict.fromkeys(resolved_compare))

    limit = merged.get("limit")
    if limit is not None:
        try:
            merged["limit"] = int(limit)
        except (TypeError, ValueError):
            merged["limit"] = None

    plan["csv_query_spec"] = merged
    return plan


# ══════════════════════════════════════════════════════════════════════════════
# Stage runners
# ══════════════════════════════════════════════════════════════════════════════

def _run_classifier(
    user_query: str,
    llm: LocalLLM,
    system_prompt: str,
    history_block: str,
    retry_once: bool = True,
) -> dict | None:
    """
    Stage 1: classify intent.
    Returns a dict with intent/scholarships/reason, or None on total failure.
    use_json_format=True forces Ollama to produce valid JSON at inference level.
    """
    user_prompt = (
        f"User query:\n{user_query}\n\n"
        f"Recent conversation:\n{history_block}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    # max_new_tokens=150 is enough for the small Stage 1 output
    raw    = llm.chat(messages, max_new_tokens=150, temperature=0.0, use_json_format=True)
    result = _extract_json(raw)
    if result and result.get("intent") in VALID_INTENTS:
        return result

    if retry_once:
        repair = (
            "\n\nYour previous output was invalid. "
            "Return ONLY the JSON object with 'intent', 'scholarships', and 'reason'. "
            "No explanation, no markdown, nothing else."
        )
        messages_r = [
            {"role": "system", "content": system_prompt + repair},
            {"role": "user",   "content": user_prompt},
        ]
        raw2    = llm.chat(messages_r, max_new_tokens=150, temperature=0.0, use_json_format=True)
        result2 = _extract_json(raw2)
        if result2 and result2.get("intent") in VALID_INTENTS:
            return result2

    return None


def _run_csv_extractor(
    user_query: str,
    llm: LocalLLM,
    df: pd.DataFrame,
    retry_once: bool = True,
) -> dict:
    """
    Stage 2a: extract CSV query spec.
    Only called when Stage 1 returned CSV intent.
    Returns a spec dict (possibly the default skeleton on failure).
    """
    system_prompt = _build_csv_extractor_prompt(df)
    user_prompt   = f"Extract the CSV query spec for this query:\n{user_query}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    raw    = llm.chat(messages, max_new_tokens=200, temperature=0.0, use_json_format=True)
    result = _extract_json(raw)
    if isinstance(result, dict):
        return result

    if retry_once:
        repair = (
            "\n\nReturn ONLY the JSON spec object. "
            "No explanation, no markdown fences, no preamble."
        )
        messages_r = [
            {"role": "system", "content": system_prompt + repair},
            {"role": "user",   "content": user_prompt},
        ]
        raw2    = llm.chat(messages_r, max_new_tokens=200, temperature=0.0, use_json_format=True)
        result2 = _extract_json(raw2)
        if isinstance(result2, dict):
            return result2

    print("[router] CSV extractor failed both attempts — using default rows spec")
    return {}  # validate_router_output will fill defaults


def _run_profile_extractor(
    user_query: str,
    llm: LocalLLM,
    retry_once: bool = True,
) -> dict:
    """
    Stage 2b: extract profile fields.
    Only called when Stage 1 returned PROFILE intent.
    Returns a field dict (possibly empty on failure).
    """
    system_prompt = _build_profile_extractor_prompt()
    user_prompt   = f"Extract profile fields from:\n{user_query}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    raw    = llm.chat(messages, max_new_tokens=100, temperature=0.0, use_json_format=True)
    result = _extract_json(raw)
    if isinstance(result, dict):
        return result

    if retry_once:
        repair = (
            "\n\nReturn ONLY the JSON object with extracted fields. "
            "If nothing found, return {}. No explanation."
        )
        messages_r = [
            {"role": "system", "content": system_prompt + repair},
            {"role": "user",   "content": user_prompt},
        ]
        raw2    = llm.chat(messages_r, max_new_tokens=100, temperature=0.0, use_json_format=True)
        result2 = _extract_json(raw2)
        if isinstance(result2, dict):
            return result2

    print("[router] Profile extractor failed both attempts — returning empty dict")
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════════

def route_query(
    user_query: str,
    llm: LocalLLM,
    df: pd.DataFrame,
    name_list: list[str],
    recent_history: list[dict] | None = None,
    last_scholarships: list[str] | None = None,
    retry_once: bool = True,
) -> dict[str, Any]:
    """
    2-stage routing pipeline.

    Stage 1 — intent classification (always runs, ~150 output tokens):
      Small, focused prompt. Returns intent + scholarship names only.
      No nested schema, no filter extraction, no profile parsing.

    Stage 2 — schema extraction (conditional, only for CSV and PROFILE):
      CSV route    → _run_csv_extractor(): extracts filters, operation, sort, etc.
      PROFILE route → _run_profile_extractor(): extracts field:value pairs.
      RAG / TXT_ONLY / GREETING / FAREWELL / OFF_TOPIC: no Stage 2 call needed.

    Why this is better than the original single-pass approach:
      A 7B model asked to do 6 things simultaneously will drop fields, confuse
      keys, and hallucinate schema structure at significant rates even at
      temperature=0.0. Splitting into focused single-task calls improves
      per-stage reliability substantially, and the total latency cost for
      simple routes (RAG, GREETING, etc.) is *lower* because Stage 1 alone
      is much cheaper than the original combined prompt.

    Parameters
    ----------
    user_query        : current user message
    llm               : LocalLLM instance
    df                : live scholarships dataframe
    name_list         : canonical scholarship name list
    recent_history    : list[{"role": "user"|"assistant", "message": str}]
    last_scholarships : scholarship names from the most recent assistant turn
    retry_once        : retry each stage once on failure
    """
    # ── Build history block (shared across stages) ─────────────────────────────
    history_lines: list[str] = []
    for h in (recent_history or [])[-4:]:
        role_label = "User" if h.get("role") == "user" else "Assistant"
        msg = (h.get("message") or "")[:300]
        history_lines.append(f"  {role_label}: {msg}")
    history_block = "\n".join(history_lines) or "  (none)"

    # ── Stage 1: classify intent ───────────────────────────────────────────────
    classifier_prompt = _build_classifier_prompt(df, name_list, last_scholarships)
    stage1 = _run_classifier(
        user_query, llm, classifier_prompt, history_block, retry_once
    )

    if stage1 is None:
        print("[router] Stage 1 classification failed — falling back to RAG")
        stage1 = {"intent": "RAG", "scholarships": [], "reason": "classifier fallback"}

    intent    = stage1.get("intent", "RAG")
    raw_names = stage1.get("scholarships") or []
    reason    = stage1.get("reason", "")
    route     = _INTENT_TO_ROUTE.get(intent, "RAG")

    # ── Stage 2: schema extraction (only when needed) ──────────────────────────
    csv_spec:       dict = {}
    profile_update: dict = {}

    if route == "CSV_ONLY":
        csv_spec = _run_csv_extractor(user_query, llm, df, retry_once)

    elif route == "PROFILE_UPDATE":
        profile_update = _run_profile_extractor(user_query, llm, retry_once)

    # ── Assemble and validate full plan ───────────────────────────────────────
    plan: dict[str, Any] = {
        "route":          route,
        "scholarships":   raw_names,
        "csv_query_spec": csv_spec,
        "profile_update": profile_update,
        "reason":         reason,
    }

    try:
        return validate_router_output(plan, df, name_list)
    except ValueError as e:
        print(f"[router] Validation failed after stage 2: {e} — falling back to RAG")
        fallback: dict[str, Any] = {
            "route":          "RAG",
            "scholarships":   [],
            "csv_query_spec": {},
            "profile_update": {},
            "reason":         f"validation fallback: {e}",
        }
        return validate_router_output(fallback, df, name_list)