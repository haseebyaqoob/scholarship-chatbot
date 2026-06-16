

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import pandas as pd

from config_loader import cfg
from database import DatabaseManager, DB_PATH, FIELD_LABELS
from llm_handler import LocalLLM
from router import route_query, validate_router_output
from scholarship_index import ScholarshipIndex, RAGEngine

_ANSWER_TOKENS = int(cfg["answer_max_tokens"])
_RAG_TOP_K     = int(cfg["rag_top_k"])

_SCHOLARSHIP_SIGNALS = frozenset({
    "scholarship", "scholarships", "fellowship", "fellowships",
    "apply", "application", "eligible", "eligibility",
    "qualify", "qualification", "funding", "fund",
    "grant", "stipend", "deadline", "recommend", "suggest",
    "show me", "find me", "which scholarship", "what scholarship",
    "how many scholarship",
})

_GREETING_TOKENS = frozenset({
    "hi", "hello", "hey", "hiya", "howdy",
    "salam", "salaam", "assalam", "assalamu", "asslam",
    "greetings", "sup", "yo",
})
_FAREWELL_TOKENS = frozenset({
    "bye", "goodbye", "cya", "later", "thanks",
    "thank", "thankyou", "thanku", "cheers",
})
_FAREWELL_PHRASES = frozenset({
    "see you", "take care", "good night", "good bye",
    "thank you", "many thanks", "much appreciated", "have a good",
})


# ══════════════════════════════════════════════════════════════════════════════
# Deterministic pre-router
# ══════════════════════════════════════════════════════════════════════════════

def _deterministic_pre_route(query: str) -> str | None:
    stripped = query.strip()
    if not stripped:
        return "EMPTY"

    lower = stripped.lower()
    clean = re.sub(r"[^\w\s]", "", lower).strip()
    words = clean.split()

    if not words:
        return "EMPTY"

    if len(words) <= 3:
        if all(w in _GREETING_TOKENS for w in words):
            return "GREETING"
        if any(w in _FAREWELL_TOKENS for w in words):
            return "FAREWELL"
        if any(phrase in lower for phrase in _FAREWELL_PHRASES):
            return "FAREWELL"

    return None


# ══════════════════════════════════════════════════════════════════════════════
# CSV query execution
# ══════════════════════════════════════════════════════════════════════════════

def run_csv_query(spec: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    """
    Execute a structured CSV query spec deterministically against the dataframe.

    Fix vs previous version:
      The `fields` column-selection step previously ran before the operation
      dispatch. This caused the compare operation to crash with
      "['name'] not in index" because the name column had been dropped by
      field selection before the compare branch tried to filter by it.

      Fix: `fields` selection is now applied only inside the 'rows' branch
      (the only branch that legitimately needs column narrowing at output time).
      All other branches (compare, aggregate, etc.) work on the full
      filtered/sorted dataframe and select their own columns internally.
    """
    data = df.copy()

    # ── Apply filters ──────────────────────────────────────────────────────────
    filters = spec.get("filters") or {}

    scholarship_names = filters.get("scholarship_names") or filters.get("name") or []
    if isinstance(scholarship_names, str):
        scholarship_names = [scholarship_names]
    if scholarship_names and "name" in data.columns:
        data = data[data["name"].isin(scholarship_names)]

    skip_keys = {"scholarship_names", "name"}
    for key, val in filters.items():
        if key in skip_keys or val is None:
            continue
        if key not in data.columns:
            continue
        if isinstance(val, list):
            data = data[data[key].isin(val)]
        elif isinstance(val, dict):
            for op, operand in val.items():
                if operand is None:
                    continue
                try:
                    if op in ("gte", "ge"):
                        data = data[data[key] >= operand]
                    elif op in ("gt",):
                        data = data[data[key] > operand]
                    elif op in ("lte", "le"):
                        data = data[data[key] <= operand]
                    elif op in ("lt",):
                        data = data[data[key] < operand]
                    elif op in ("eq",):
                        data = data[data[key] == operand]
                except Exception:
                    pass
        else:
            if pd.api.types.is_string_dtype(data[key]):
                data = data[data[key].astype(str).str.lower() == str(val).lower()]
            else:
                try:
                    data = data[data[key] == type(data[key].iloc[0])(val)]
                except Exception:
                    data = data[data[key] == val]

    # Legacy shorthands
    if "min_gpa_gte" in filters and filters["min_gpa_gte"] is not None and "gpa_min" in data.columns:
        data = data[data["gpa_min"] >= float(filters["min_gpa_gte"])]
    if "max_gpa_lte" in filters and filters["max_gpa_lte"] is not None and "gpa_min" in data.columns:
        data = data[data["gpa_min"] <= float(filters["max_gpa_lte"])]

    if "deadline_before" in filters and filters["deadline_before"] and "deadline" in data.columns:
        try:
            data = data[data["deadline"] <= pd.to_datetime(filters["deadline_before"])]
        except Exception:
            pass
    if "deadline_after" in filters and filters["deadline_after"] and "deadline" in data.columns:
        try:
            data = data[data["deadline"] >= pd.to_datetime(filters["deadline_after"])]
        except Exception:
            pass

    # ── Sort ───────────────────────────────────────────────────────────────────
    sort_specs = spec.get("sort") or []
    if sort_specs:
        sort_cols = [s["field"] for s in sort_specs if s.get("field") in data.columns]
        ascending = [
            s.get("direction", "asc") == "asc"
            for s in sort_specs if s.get("field") in data.columns
        ]
        if sort_cols:
            data = data.sort_values(sort_cols, ascending=ascending)

    # ── Limit ──────────────────────────────────────────────────────────────────
    limit = spec.get("limit")
    if limit:
        try:
            data = data.head(int(limit))
        except Exception:
            pass

    # NOTE: `fields` column selection intentionally NOT applied here.
    # It was moved inside the 'rows' branch below to avoid stripping columns
    # (especially 'name') that other operation branches need.

    data      = data.reset_index(drop=True)
    operation = spec.get("operation", "rows")

    # ── Operation dispatch ─────────────────────────────────────────────────────
    try:
        if operation == "count":
            return {"ok": True, "operation": "count", "count": len(data)}

        elif operation == "exists":
            return {
                "ok": True, "operation": "exists",
                "exists": len(data) > 0, "count": len(data),
            }

        elif operation == "distinct":
            field = spec.get("field")
            if not field or field not in data.columns:
                return {"ok": False, "error": f"distinct: field '{field}' not found in dataframe"}
            values = data[field].dropna().unique().tolist()
            return {"ok": True, "operation": "distinct", "field": field, "values": values}

        elif operation == "aggregate":
            agg_spec  = spec.get("aggregate") or {}
            agg_type  = agg_spec.get("type")
            agg_field = agg_spec.get("field")
            if not agg_field or agg_field not in data.columns:
                return {"ok": False, "error": f"aggregate: field '{agg_field}' not available"}
            col = data[agg_field].dropna()
            if agg_type == "min":
                val = col.min()
            elif agg_type == "max":
                val = col.max()
            elif agg_type == "avg":
                val = col.mean()
            elif agg_type == "sum":
                val = col.sum()
            else:
                return {"ok": False, "error": f"Unknown aggregate type: {agg_type}"}
            if hasattr(val, "strftime"):
                val = val.strftime("%d %B %Y")
            return {
                "ok": True, "operation": "aggregate",
                "aggregate": agg_type, "field": agg_field, "value": val,
            }

        elif operation == "compare":
            # compare always works from the full filtered dataframe — never from
            # a field-narrowed subset — so 'name' is guaranteed to be present.
            compare_names  = spec.get("compare_names") or []
            compare_fields = spec.get("compare_fields") or []
            if not compare_names:
                compare_names = data["name"].tolist() if "name" in data.columns else []
            if compare_names and "name" in data.columns:
                data = data[data["name"].isin(compare_names)]
            if compare_fields:
                # Always include 'name' first so the output is readable
                available = (
                    (["name"] if "name" in data.columns else [])
                    + [f for f in compare_fields if f in data.columns and f != "name"]
                )
                if available:
                    data = data[available]
            return _rows_result(data)

        elif operation == "group_by":
            group_field = spec.get("group_field")
            if not group_field or group_field not in data.columns:
                return {"ok": False, "error": f"group_by: field '{group_field}' not in dataframe"}
            grouped = data.groupby(group_field).size().reset_index(name="count")
            return _rows_result(grouped)

        else:  # "rows" — apply field selection here, not before dispatch
            fields = [f for f in (spec.get("fields") or []) if f in data.columns]
            if fields:
                data = data[fields]
            return _rows_result(data)

    except Exception as e:
        return {"ok": False, "error": str(e)}


def _rows_result(data: pd.DataFrame) -> dict[str, Any]:
    out = data.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%d %B %Y").where(out[col].notna(), other=None)
    rows = out.to_dict(orient="records")
    return {"ok": True, "operation": "rows", "count": len(rows), "rows": rows}


def _summarise_csv_result(result: dict[str, Any], df: pd.DataFrame | None = None) -> str:
    """
    Format a CSV query result for display.

    Fix vs previous version:
      The friendly-label map previously hardcoded column names like "gpa_min",
      "amount_pkr", "funding_type". If the CSV used different names, every
      column showed as unlabelled.
      Fix: any column not in the friendly-label map is shown with its actual
      column name as the label (title-cased), so no data is ever silently hidden.
    """
    if not result.get("ok"):
        return f"CSV query error: {result.get('error', 'unknown')}"

    op = result.get("operation")
    if op == "count":
        return f"Total matching scholarships: {result['count']}"
    if op == "exists":
        return f"{'Yes' if result['exists'] else 'No'} — found {result['count']} matching scholarship(s)."
    if op == "distinct":
        vals = result.get("values", [])
        return f"Distinct values for '{result.get('field')}': {', '.join(str(v) for v in vals)}"
    if op == "aggregate":
        return (
            f"{result.get('aggregate', '').upper()} of '{result.get('field')}': "
            f"{result.get('value')}"
        )

    rows = result.get("rows", [])
    if not rows:
        return "No scholarships found matching your criteria."

    # Friendly labels for well-known column names; anything else uses title-case name
    known_labels: dict[str, str] = {
        "level":        "Level",
        "field":        "Field",
        "domicile":     "Domicile",
        "funding_type": "Funding",
        "gpa_min":      "Min GPA",
        "cgpa":         "Min CGPA",
        "gpa":          "GPA",
        "amount_pkr":   "Amount",
        "deadline":     "Deadline",
        "status":       "Status",
        "country":      "Country",
    }

    lines = []
    for row in rows:
        name   = row.get("name", "")
        header = f"**{name}**" if name else "Scholarship"
        details = []
        for col, val in row.items():
            if col == "name":
                continue
            if val in (None, "", "nan", float("nan")):
                continue
            label = known_labels.get(col, col.replace("_", " ").title())
            if col == "amount_pkr":
                try:
                    val = f"PKR {int(float(val)):,}"
                except Exception:
                    pass
            details.append(f"{label}: {val}")
        lines.append(f"{header}\n  " + " | ".join(details))
    return "\n\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Answer LLM system prompt
# ══════════════════════════════════════════════════════════════════════════════

def _build_answer_system(name_list: list[str], profile: dict) -> str:
    today   = datetime.now().strftime("%d %B %Y")
    catalog = "\n".join(f"  - {n}" for n in name_list)
    profile_lines = []
    for k, label in FIELD_LABELS.items():
        v = profile.get(k)
        if v and str(v) not in ("Student", "None", ""):
            profile_lines.append(f"  {label}: {v}")
    profile_block = (
        ("Current student profile:\n" + "\n".join(profile_lines))
        if profile_lines else ""
    )

    return f"""You are a scholarship assistant for university students in Pakistan.
Today's date: {today}

{profile_block}

━━━ SCHOLARSHIPS YOU ARE ALLOWED TO DISCUSS ━━━
{catalog}

━━━ ABSOLUTE RULES — READ CAREFULLY ━━━

You have NO knowledge beyond what is written below in the evidence section.
Even if you believe you know about a scholarship from your training data,
that information does NOT exist for you here. You must treat your training
knowledge as completely inaccessible.

- Answer ONLY from the evidence provided below.
- If the evidence does not contain the answer, say clearly:
  "I don't have that information in my database."
- NEVER invent, guess, or extrapolate facts about any scholarship.
- NEVER discuss a scholarship that is not in the list above.
- NEVER use outside knowledge about any scholarship you may have seen during training.
- If a deadline has passed (relative to today's date), flag it explicitly.
- Keep answers concise, factual, and structured with bullet points where helpful.
- List ALL scholarships from the evidence — do NOT omit any.
- If the student profile is available, tailor your answer to their background.

━━━ PROFILE QUESTIONS ━━━
If the user asks about their own profile ("what is my name", "what's my GPA",
"do you remember my details"), answer from the "Current student profile" block
above. Do not say you don't have that information if it is in the profile block.
"""


# ══════════════════════════════════════════════════════════════════════════════
# execute_plan
# ══════════════════════════════════════════════════════════════════════════════

def execute_plan(
    plan: dict[str, Any],
    user_query: str,
    sch_index: ScholarshipIndex,
    rag_engine: RAGEngine,
    profile: dict,
) -> dict[str, Any]:
    route           = plan["route"]
    scholarships    = plan.get("scholarships") or []
    csv_spec        = plan.get("csv_query_spec") or {}
    evidence_parts: list[dict] = []
    execution_log:  list[dict] = []
    direct_answer:  str | None = None

    if route == "CSV_ONLY":
        csv_result = run_csv_query(csv_spec, sch_index.df)
        execution_log.append({
            "step":      "run_csv_query",
            "ok":        csv_result.get("ok"),
            "operation": csv_result.get("operation"),
            "count":     csv_result.get("count", 0),
        })
        summary       = _summarise_csv_result(csv_result, sch_index.df)
        direct_answer = summary
        evidence_parts.append({"source_type": "csv", "content": summary, "raw": csv_result})

    elif route == "TXT_ONLY":
        for name in scholarships:
            match = sch_index.fuzzy_find(name)
            if not match:
                continue
            row = match["row"]
            txt = sch_index.load_txt(row)
            if txt:
                evidence_parts.append({
                    "source_type": "txt",
                    "content":     f"[{row['name']}]\n{txt}",
                    "raw":         {"name": row["name"]},
                })
                execution_log.append({"step": "load_txt", "name": row["name"], "ok": True})
            else:
                summary = sch_index.to_summary_string(row)
                evidence_parts.append({
                    "source_type": "csv_fallback",
                    "content":     f"[{row['name']}] (no detail file)\n{summary}",
                    "raw":         {},
                })
                execution_log.append({
                    "step": "load_txt", "name": row["name"],
                    "ok": False, "note": "fallback to CSV",
                })

    elif route == "RAG":
        if csv_spec.get("filters"):
            filtered_df  = sch_index.filter_by_attributes(
                _csv_spec_to_attr_filters(csv_spec, sch_index.df)
            )
            filter_names = filtered_df["name"].tolist() if not filtered_df.empty else None
        elif scholarships:
            filter_names = scholarships
        else:
            filter_names = None

        hits = rag_engine.retrieve(user_query, top_k=_RAG_TOP_K, scholarship_names=filter_names)
        execution_log.append({
            "step":         "rag_retrieve",
            "hits":         len(hits),
            "filter_names": filter_names,
            "hybrid_search": True,
        })

        # ── No-evidence message (used by both guards below) ────────────────────
        _NO_EVIDENCE_MSG = (
            "I don't have detailed information about that in my database. "
            "Try asking about a specific scholarship by name, or rephrase your question."
        )

        if not hits:
            # No chunks passed the hybrid score threshold — do NOT send CSV
            # summaries as evidence; that causes hallucination.
            direct_answer = _NO_EVIDENCE_MSG
            execution_log.append({
                "step": "rag_no_hits",
                "note": "no chunks above min_score threshold — LLM skipped",
            })
        else:
            # ── Sparse-evidence safeguard ──────────────────────────────────────
            # If the total retrieved content is too thin (< 500 chars) the LLM
            # has nothing meaningful to ground on and will hallucinate details.
            # Return a direct answer instead of calling the LLM.
            total_content_len = sum(len(h["content"]) for h in hits)
            if total_content_len < 500:
                direct_answer = _NO_EVIDENCE_MSG
                execution_log.append({
                    "step":      "rag_sparse_evidence",
                    "note":      f"total content {total_content_len} chars < 500 threshold",
                    "triggered": True,
                })
            else:
                by_name: dict[str, list[str]] = {}
                for hit in hits:
                    by_name.setdefault(hit["scholarship_name"], []).append(hit["content"])
                for name, chunks in by_name.items():
                    evidence_parts.append({
                        "source_type": "rag",
                        "content":     f"[{name}]\n" + "\n---\n".join(chunks),
                        "raw":         {"name": name},
                    })

    return {
        "evidence_parts": evidence_parts,
        "execution_log":  execution_log,
        "direct_answer":  direct_answer,
    }


def _csv_spec_to_attr_filters(csv_spec: dict, df: pd.DataFrame) -> dict:
    filters = csv_spec.get("filters") or {}
    attr: dict = {}

    column_map = {
        "level":        "level",
        "field":        "field",
        "domicile":     "domicile",
        "funding_type": "funding_type",
    }

    for csv_col, attr_key in column_map.items():
        val = filters.get(csv_col)
        if val is not None:
            attr[attr_key] = val

    # Handle GPA column generically — find whichever numeric column is GPA-related
    gpa_col = next(
        (c for c in df.columns
         if pd.api.types.is_numeric_dtype(df[c])
         and any(kw in c.lower() for kw in ("gpa", "cgpa", "grade"))),
        None,
    )
    if gpa_col:
        gpa_val = filters.get(gpa_col) or filters.get("gpa_min") or filters.get("cgpa")
        if gpa_val is not None:
            if isinstance(gpa_val, dict):
                extracted = (
                    gpa_val.get("lte") or gpa_val.get("le") or gpa_val.get("lt")
                )
                if extracted is not None:
                    attr["min_gpa"] = extracted
            else:
                try:
                    attr["min_gpa"] = float(gpa_val)
                except (TypeError, ValueError):
                    pass

    return attr


# ══════════════════════════════════════════════════════════════════════════════
# Answer generator
# ══════════════════════════════════════════════════════════════════════════════

def generate_final_answer(
    user_query: str,
    plan: dict[str, Any],
    execution: dict[str, Any],
    name_list: list[str],
    profile: dict,
    llm: LocalLLM,
    history: list[dict],
) -> str:
    if execution.get("direct_answer"):
        return execution["direct_answer"]

    evidence_parts = execution["evidence_parts"]
    if not evidence_parts:
        return (
            "I couldn't find relevant information for your query. "
            "Try rephrasing or ask about a specific scholarship."
        )

    evidence_text = "\n\n".join(
        f"Source {i+1} ({p['source_type']}):\n{p['content']}"
        for i, p in enumerate(evidence_parts)
    )

    system   = _build_answer_system(name_list, profile)
    messages = [{"role": "system", "content": system}]
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["message"]})
    messages.append({
        "role": "user",
        "content": (
            f"{user_query}\n\n"
            f"---\n[EVIDENCE — answer strictly from this]\n\n"
            f"{evidence_text}\n---"
        ),
    })

    try:
        return llm.chat(messages, max_new_tokens=_ANSWER_TOKENS, temperature=0.0)
    except Exception as e:
        return f"⚠️ Generation error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# ScholarshipPipeline
# ══════════════════════════════════════════════════════════════════════════════

class ScholarshipPipeline:
    def __init__(self, llm: LocalLLM):
        self.llm     = llm
        self.db      = DatabaseManager(DB_PATH)
        self.idx     = ScholarshipIndex()
        self.rag     = RAGEngine(self.idx)
        self.session = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"[pipeline] Ready — {len(self.idx.name_list)} scholarships | session: {self.session}")

    def set_session(self, session_id: str):
        self.session = session_id

    # ── Confirmation helpers ───────────────────────────────────────────────────

    @staticmethod
    def _is_affirmative(text: str) -> bool:
        t     = text.lower().strip()
        words = t.split()
        if len(words) > 5:
            return False
        return any(w in t for w in [
            "yes", "yeah", "yep", "sure", "ok", "okay",
            "confirm", "update it", "go ahead", "do it",
            "change it", "please do",
        ])

    @staticmethod
    def _is_negative(text: str) -> bool:
        t = text.lower().strip()
        return any(w in t for w in [
            "no", "nope", "cancel", "keep", "don't", "dont",
            "skip", "leave it", "ignore", "never mind",
        ])

    @staticmethod
    def _has_scholarship_intent(query: str) -> bool:
        t = query.lower()
        return any(sig in t for sig in _SCHOLARSHIP_SIGNALS)

    @staticmethod
    def _build_confirm_prompt(pending: dict, current: dict) -> str:
        lines = []
        for field, new_val in pending.items():
            old_val = current.get(field)
            label   = FIELD_LABELS.get(field, field.title())
            if old_val and str(old_val) not in ("Student", "None", ""):
                lines.append(f"  • {label}: '{old_val}' → '{new_val}'")
            else:
                lines.append(f"  • {label}: set to '{new_val}'")
        changes = "\n".join(lines)
        return (
            f"I'd like to update your profile:\n\n"
            f"{changes}\n\n"
            f"Shall I apply these changes? (yes / no)"
        )

    @staticmethod
    def _name_echo(name: str) -> str:
        return f"\n\n*(Saved your name as **{name}** — if that's not right, just correct me.)*"

    # ── Main query handler ─────────────────────────────────────────────────────

    def process_query(self, query: str) -> str:
        query = query.strip()

        # ── Step 0: deterministic pre-routing ─────────────────────────────────
        pre_route = _deterministic_pre_route(query)

        if pre_route == "EMPTY":
            return "Please type a question!"

        if pre_route in ("GREETING", "FAREWELL"):
            profile     = self.db.get_profile()
            stored_name = profile.get("name", "")

            if pre_route == "GREETING":
                display  = stored_name if stored_name and stored_name != "Student" else "there"
                response = (
                    f"Hello {display}! 👋 I'm your Scholarship Assistant.\n\n"
                    "I can help you:\n"
                    "• Find scholarships matching your background\n"
                    "• Check deadlines and eligibility criteria\n"
                    "• Understand application requirements\n\n"
                    'Try: *"Scholarships for postgraduate students in Sindh"* or '
                    '*"What is the HEC deadline?"*'
                )
            else:
                name_bit = f" {stored_name}" if stored_name and stored_name != "Student" else ""
                if any(w in query.lower() for w in ["bye", "goodbye", "see you", "take care", "cya"]):
                    response = (
                        f"Goodbye{name_bit}! 👋 Best of luck with your scholarship applications. "
                        "Feel free to come back anytime!"
                    )
                else:
                    response = (
                        f"You're welcome{name_bit}! 😊 "
                        "Let me know if you have any other scholarship questions."
                    )

            self.db.save_message(self.session, "user",      query)
            self.db.save_message(self.session, "assistant", response, pre_route.lower())
            return response

        # ── Step 1: handle pending confirmation ───────────────────────────────
        profile = self.db.get_profile()
        pending = self.db.get_pending_updates()

        if pending:
            if self._is_affirmative(query):
                self.db.update_profile(**pending)
                self.db.clear_pending_updates()
                updated  = self.db.get_profile()
                disp     = updated.get("name", "")
                greet    = f", {disp}" if disp and disp != "Student" else ""
                response = (
                    f"✅ Done{greet}! Profile updated:\n\n"
                    + "\n".join(
                        f"  • {FIELD_LABELS.get(k, k)}: {v}" for k, v in pending.items()
                    )
                    + "\n\nHow can I help you with scholarships?"
                )
                self.db.save_message(self.session, "user",      query)
                self.db.save_message(self.session, "assistant", response, "profile_confirm")
                return response

            elif self._is_negative(query):
                self.db.clear_pending_updates()
                name  = profile.get("name", "")
                greet = f" {name}" if name and name != "Student" else ""
                response = (
                    f"No problem{greet}! Your profile stays unchanged. "
                    "What would you like to explore?"
                )
                self.db.save_message(self.session, "user",      query)
                self.db.save_message(self.session, "assistant", response, "profile_deny")
                return response

            else:
                _words     = query.strip().split()
                _looks_new = (
                    len(_words) > 6
                    or "?" in query
                    or self._has_scholarship_intent(query)
                )
                if _looks_new:
                    self.db.clear_pending_updates()
                    pending = {}
                else:
                    response = (
                        "I still have a pending profile update waiting:\n\n"
                        + self._build_confirm_prompt(pending, profile)
                    )
                    self.db.save_message(self.session, "user",      query)
                    self.db.save_message(self.session, "assistant", response, "profile_confirm_reprompt")
                    return response

        # ── Step 2: gather context ────────────────────────────────────────────
        history           = self.db.get_recent_history(self.session, n=3)
        last_scholarships = self.db.get_last_scholarships(self.session)
        self.db.save_message(self.session, "user", query)

        # ── Step 3: route ──────────────────────────────────────────────────────
        try:
            plan = route_query(
                query,
                self.llm,
                self.idx.df,
                self.idx.name_list,
                recent_history=history,
                last_scholarships=last_scholarships,
            )
        except Exception as e:
            return f"⚠️ Routing error: {e}"

        route = plan["route"]

        # ── Step 4: conversational routes ──────────────────────────────────────
        if route == "GREETING":
            stored_name = profile.get("name", "")
            display     = stored_name if stored_name and stored_name != "Student" else "there"
            response = (
                f"Hello {display}! 👋 I'm your Scholarship Assistant.\n\n"
                "I can help you:\n"
                "• Find scholarships matching your background\n"
                "• Check deadlines and eligibility criteria\n"
                "• Understand application requirements\n\n"
                'Try: *"Scholarships for postgraduate students in Sindh"* or '
                '*"What is the HEC deadline?"*'
            )
            self.db.save_message(self.session, "assistant", response, "greeting")
            return response

        if route == "FAREWELL":
            stored_name = profile.get("name", "")
            name_bit    = f" {stored_name}" if stored_name and stored_name != "Student" else ""
            if any(w in query.lower() for w in ["bye", "goodbye", "see you", "take care", "cya"]):
                response = (
                    f"Goodbye{name_bit}! 👋 Best of luck with your scholarship applications. "
                    "Feel free to come back anytime!"
                )
            else:
                response = (
                    f"You're welcome{name_bit}! 😊 "
                    "Let me know if you have any other scholarship questions."
                )
            self.db.save_message(self.session, "assistant", response, "farewell")
            return response

        if route == "OFF_TOPIC":
            stored_name = profile.get("name", "")
            name_bit    = f", {stored_name}" if stored_name and stored_name != "Student" else ""
            response = (
                f"I'm only able to help with scholarship-related questions{name_bit}. 🎓\n\n"
                "Feel free to ask me about:\n"
                "• Scholarships matching your academic background\n"
                "• Eligibility criteria and GPA requirements\n"
                "• Application deadlines and processes\n\n"
                "What would you like to know about scholarships?"
            )
            self.db.save_message(self.session, "assistant", response, "off_topic")
            return response

        if route == "PROFILE_UPDATE":
            updates = plan.get("profile_update") or {}
            if not updates:
                response = (
                    "Could you share more details? You can tell me your:\n"
                    "name, GPA, field of study, academic level, province, or nationality."
                )
                self.db.save_message(self.session, "assistant", response, "profile_update_empty")
                return response

            overwrite_fields: dict = {}
            new_fields:       dict = {}
            for field, new_val in updates.items():
                old_val = profile.get(field)
                if old_val and str(old_val) not in ("Student", "None", "", "nan"):
                    overwrite_fields[field] = new_val
                else:
                    new_fields[field] = new_val

            if new_fields:
                self.db.update_profile(**new_fields)

            if overwrite_fields:
                self.db.set_pending_updates(overwrite_fields)
                confirm_msg = self._build_confirm_prompt(overwrite_fields, profile)
                if new_fields:
                    new_summary = "\n".join(
                        f"  • {FIELD_LABELS.get(k, k)}: {v}" for k, v in new_fields.items()
                    )
                    response = f"✅ Noted:\n{new_summary}\n\nOne more thing:\n\n{confirm_msg}"
                elif "name" in overwrite_fields and len(overwrite_fields) == 1:
                    old_name = profile.get("name", "")
                    new_name = overwrite_fields["name"]
                    response = (
                        f"Your profile currently has your name as **{old_name}**. "
                        f"Would you like me to update it to **{new_name}**? (yes / no)"
                    )
                else:
                    response = confirm_msg
            else:
                profile  = self.db.get_profile()
                name     = profile.get("name", "")
                name_str = f", {name}" if name and name != "Student" else ""

                if list(new_fields.keys()) == ["name"]:
                    response = (
                        f"Nice to meet you, {name}! 👋 I've saved your name.\n\n"
                        "You can also share your GPA, field of study, province, or nationality "
                        "so I can personalise scholarship recommendations for you."
                    )
                else:
                    fields_summary = "\n".join(
                        f"  • {FIELD_LABELS.get(k, k)}: {v}" for k, v in new_fields.items()
                    )
                    response = (
                        f"✅ Got it{name_str}! I've noted:\n\n"
                        f"{fields_summary}\n\n"
                        "I'll use this to personalise your recommendations. "
                        "What would you like to explore?"
                    )
                    if "name" in new_fields:
                        response += self._name_echo(new_fields["name"])

            # Mixed-intent check
            if not overwrite_fields and self._has_scholarship_intent(query):
                try:
                    sch_plan = validate_router_output(
                        {
                            "route":          "RAG",
                            "scholarships":   [],
                            "csv_query_spec": {},
                            "profile_update": {},
                            "reason":         "mixed intent",
                        },
                        self.idx.df,
                        self.idx.name_list,
                    )
                    updated_profile = self.db.get_profile()
                    sch_execution   = execute_plan(
                        sch_plan, query, self.idx, self.rag, updated_profile
                    )
                    sch_answer = generate_final_answer(
                        query, sch_plan, sch_execution,
                        self.idx.name_list, updated_profile, self.llm, history,
                    )
                    response = response + "\n\n---\n\n" + sch_answer
                    referenced = list({
                        p["raw"].get("name", "")
                        for p in sch_execution["evidence_parts"]
                        if isinstance(p.get("raw"), dict) and p["raw"].get("name")
                    })
                    self.db.save_message(
                        self.session, "assistant", response,
                        "profile_update_with_rag", referenced,
                    )
                    return response
                except Exception as exc:
                    print(f"[pipeline] Mixed-intent lookup failed: {exc}")

            self.db.save_message(self.session, "assistant", response, "profile_update")
            return response

        # ── Step 5: execute plan ───────────────────────────────────────────────
        try:
            execution = execute_plan(plan, query, self.idx, self.rag, profile)
        except Exception as e:
            return f"⚠️ Execution error: {e}"

        # ── Step 6: generate answer ────────────────────────────────────────────
        try:
            response = generate_final_answer(
                query, plan, execution,
                self.idx.name_list, profile,
                self.llm, history,
            )
        except Exception as e:
            response = f"⚠️ Answer generation error: {e}"

        referenced = list({
            p["raw"].get("name", "")
            for p in execution["evidence_parts"]
            if isinstance(p.get("raw"), dict) and p["raw"].get("name")
        })
        self.db.save_message(
            self.session, "assistant", response, route.lower(), referenced
        )
        return response


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_pipeline_instance: ScholarshipPipeline | None = None


def get_pipeline() -> ScholarshipPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        llm = LocalLLM()
        _pipeline_instance = ScholarshipPipeline(llm)
    return _pipeline_instance
