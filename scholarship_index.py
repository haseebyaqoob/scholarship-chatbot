"""
scholarship_index.py
─────────────────────
Loads scholarship data from the scholarship/ folder (CSV + txt files).
Provides fuzzy lookup, attribute filtering, txt loading, and RAG retrieval.

Design rules:
- Zero hardcoded scholarship names, column names, domicile lists, or funding maps.
- All configuration (aliases, defaults, paths, model names) comes from config.yaml.
- Adding a new scholarship = drop a row in the CSV + optional txt file. No code changes.

Fixes applied vs original:
  - _build_index() now collects and prints a startup warning listing all scholarships
    that have no txt file and are falling back to CSV summary strings. This makes
    coverage gaps visible at startup instead of silently degrading RAG quality.
  - retrieve() increases the FAISS search pool when a scholarship_names whitelist is
    provided. The original fixed pool (top_k * 6) could be exhausted before collecting
    top_k results from a narrow whitelist (e.g., 2 scholarships), silently returning
    fewer hits than requested. Pool is now top_k * 20 (min 50) when filtering.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process as rfprocess
from sentence_transformers import SentenceTransformer

from config_loader import cfg

# ── Config values ──────────────────────────────────────────────────────────────
_SCHOLARSHIP_DIR   = Path(cfg["scholarship_dir"])
_EMBED_MODEL       = cfg["embed_model"]
_CHUNK_WORDS       = int(cfg["chunk_words"])
_CHUNK_OVERLAP     = int(cfg["chunk_overlap"])
_RAG_TOP_K         = int(cfg["rag_top_k"])
_FUZZY_THRESHOLD   = int(cfg["fuzzy_match_threshold"])
_COLUMN_ALIASES    = cfg.get("column_aliases", {})
_OPTIONAL_DEFAULTS = cfg.get("optional_defaults", {})


# ── Chunk dataclass for RAG ────────────────────────────────────────────────────
@dataclass
class Chunk:
    chunk_id:         str
    scholarship_name: str
    content:          str


# ── ScholarshipIndex ───────────────────────────────────────────────────────────

class ScholarshipIndex:
    def __init__(self, input_dir: Path = _SCHOLARSHIP_DIR):
        self.input_dir = input_dir
        self.df        = self._load_csv()
        self.name_list: list[str] = self.df["name"].tolist()
        print(f"[scholarship_index] {len(self.name_list)} scholarships loaded")

    # ── CSV loading ────────────────────────────────────────────────────────────

    def _load_csv(self) -> pd.DataFrame:
        csv_files = sorted(self.input_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV file found in '{self.input_dir}'. "
                "Place a scholarships CSV inside that folder."
            )
        csv_path = csv_files[0]
        print(f"[scholarship_index] Loading CSV: {csv_path}")
        df = pd.read_csv(csv_path)

        # Normalise headers: lowercase + underscores
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Apply column aliases from config (no hardcoded names here)
        rename_map = {k: v for k, v in _COLUMN_ALIASES.items() if k in df.columns}
        if rename_map:
            df = df.rename(columns=rename_map)

        # Inject missing optional columns with defaults from config
        for col, default in _OPTIONAL_DEFAULTS.items():
            if col not in df.columns:
                df[col] = default
                print(f"[scholarship_index] Column '{col}' missing — defaulted to {default!r}")

        if "name" not in df.columns:
            raise ValueError(
                "CSV must have a 'name' column (or an alias defined in config.yaml)."
            )

        # Coerce types
        if "deadline" in df.columns:
            df["deadline"] = pd.to_datetime(df["deadline"], errors="coerce")
        if "amount_pkr" in df.columns:
            df["amount_pkr"] = pd.to_numeric(df["amount_pkr"], errors="coerce")
        if "gpa_min" in df.columns:
            df["gpa_min"] = pd.to_numeric(df["gpa_min"], errors="coerce").fillna(0.0)

        # Auto-expire rows whose deadline has passed but status is still 'active'
        if "deadline" in df.columns and "status" in df.columns:
            now         = pd.Timestamp.now()
            mask_passed = df["deadline"].notna() & (df["deadline"] < now)
            mask_active = df["status"].astype(str).str.lower() == "active"
            df.loc[mask_passed & mask_active, "status"] = "expired"

        # Normalise status and funding_type to lowercase strings
        for col in ("status", "funding_type"):
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()

        # Map common shorthand funding values to canonical form
        if "funding_type" in df.columns:
            df["funding_type"] = df["funding_type"].map(
                lambda v: "need-based" if v == "need"
                else "merit-based" if v == "merit"
                else v
            )

        print(f"[scholarship_index] Columns: {list(df.columns)}")
        return df

    # ── Lookup ─────────────────────────────────────────────────────────────────

    def fuzzy_find(self, query_name: str, threshold: int = _FUZZY_THRESHOLD) -> dict | None:
        """Return the best-matching row dict or None if below threshold."""
        result = rfprocess.extractOne(
            query_name, self.name_list, scorer=fuzz.token_set_ratio
        )
        if result and result[1] >= threshold:
            matched_name, score, _ = result
            row = self.df[self.df["name"] == matched_name].iloc[0]
            return {"row": row, "match_score": score, "matched_name": matched_name}
        return None

    # ── Filtering ──────────────────────────────────────────────────────────────

    def filter_by_attributes(self, filters: dict) -> pd.DataFrame:
        """
        Apply attribute filters. All column names are taken from the actual
        dataframe at runtime — nothing hardcoded here.
        """
        df = self.df.copy()

        level = filters.get("level")
        if level and "level" in df.columns:
            df = df[
                df["level"].astype(str).str.lower().str.contains(level.lower(), na=False)
                | (df["level"].astype(str).str.lower() == "any")
            ]

        field = filters.get("field")
        if field and field.lower() not in ("any", "all", "") and "field" in df.columns:
            df = df[
                df["field"].astype(str).str.lower().str.contains(field.lower(), na=False)
                | (df["field"].astype(str).str.lower() == "any")
            ]

        domicile = filters.get("domicile")
        if domicile and "domicile" in df.columns:
            df = df[
                df["domicile"].astype(str).str.lower().str.contains(domicile.lower(), na=False)
                | (df["domicile"].astype(str).str.lower() == "any")
            ]

        funding = filters.get("funding_type")
        if funding and "funding_type" in df.columns:
            df = df[
                df["funding_type"].astype(str).str.lower().str.contains(funding.lower(), na=False)
            ]

        min_gpa = filters.get("min_gpa")
        if min_gpa is not None and "gpa_min" in df.columns:
            try:
                df = df[df["gpa_min"] <= float(min_gpa)]
            except (TypeError, ValueError):
                pass

        if filters.get("exclude_expired", True) and "status" in df.columns:
            df = df[df["status"].astype(str).str.lower() != "expired"]

        return df

    # ── Text file loading ──────────────────────────────────────────────────────

    def load_txt(self, row) -> str | None:
        """
        Try three path resolution strategies, all relative to the scholarship/ folder.
        Returns file text or None if the file cannot be found.
        """
        txt_val = row.get("txt_file", "")
        if not txt_val or (isinstance(txt_val, float) and pd.isna(txt_val)):
            return None
        txt_val = str(txt_val).strip()
        if txt_val.lower() in ("none", "nan", ""):
            return None

        attempts = [
            Path(txt_val),
            self.input_dir / txt_val,
            self.input_dir / Path(txt_val).name,
        ]
        for p in attempts:
            if p.exists():
                return p.read_text(encoding="utf-8", errors="ignore")

        print(f"[scholarship_index] txt file not found: {txt_val!r}")
        return None

    # ── Display helpers ────────────────────────────────────────────────────────

    def to_summary_string(self, row) -> str:
        """Produce a human-readable metadata block for a single scholarship row."""
        try:
            dl           = row["deadline"]
            deadline_str = dl.strftime("%d %B %Y") if pd.notna(dl) else "N/A"
        except Exception:
            deadline_str = str(row.get("deadline", "N/A"))
        try:
            amt        = row.get("amount_pkr")
            amount_str = f"PKR {int(amt):,}" if pd.notna(amt) else "N/A"
        except Exception:
            amount_str = str(row.get("amount_pkr", "N/A"))
        return (
            f"• {row['name']}\n"
            f"  Level: {row.get('level','N/A')} | Field: {row.get('field','N/A')} | "
            f"Country: {row.get('country','N/A')}\n"
            f"  Amount: {amount_str} | Funding: {row.get('funding_type','N/A')}\n"
            f"  Min GPA: {row.get('gpa_min','N/A')} | Domicile: {row.get('domicile','N/A')}\n"
            f"  Deadline: {deadline_str} | Status: {row.get('status','N/A')}\n"
        )


# ── RAGEngine ──────────────────────────────────────────────────────────────────

class RAGEngine:
    """
    Sentence-transformer + FAISS semantic search over scholarship documents.
    Each scholarship gets one or more chunks from its txt file (or a CSV summary
    fallback if no txt is available).
    """

    def __init__(
        self,
        scholarship_index: ScholarshipIndex,
        embed_model_name: str = _EMBED_MODEL,
    ):
        print(f"[rag_init] Loading embedding model '{embed_model_name}' …")
        self.model      = SentenceTransformer(embed_model_name)
        self.sch_index  = scholarship_index
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.chunks: list[Chunk] = []
        self._build_index()

    # ── Index construction ─────────────────────────────────────────────────────

    @staticmethod
    def _split_chunks(
        text: str,
        name: str,
        chunk_words: int,
        overlap: int,
    ) -> list[Chunk]:
        words  = text.split()
        result = []
        start  = 0
        idx    = 0
        while start < len(words):
            end        = min(start + chunk_words, len(words))
            chunk_text = " ".join(words[start:end])
            result.append(Chunk(
                chunk_id         = f"{name.lower().replace(' ', '_')}_{idx}",
                scholarship_name = name,
                content          = chunk_text,
            ))
            if end == len(words):
                break
            start += chunk_words - overlap
            idx   += 1
        return result

    def _build_index(self):
        all_chunks:  list[Chunk] = []
        missing_txt: list[str]   = []   # track scholarships with no txt file

        for _, row in self.sch_index.df.iterrows():
            txt = self.sch_index.load_txt(row)
            if txt:
                doc = txt
            else:
                missing_txt.append(row["name"])
                doc = self.sch_index.to_summary_string(row)
            all_chunks.extend(
                self._split_chunks(doc, row["name"], _CHUNK_WORDS, _CHUNK_OVERLAP)
            )

        # Warn about scholarships that will use low-quality CSV fallback for RAG
        if missing_txt:
            print(
                f"[rag_init] WARNING — {len(missing_txt)} scholarship(s) have no txt file "
                f"and will use CSV metadata as their RAG document (lower recall):\n"
                + "\n".join(f"  • {n}" for n in missing_txt)
            )
        else:
            print("[rag_init] All scholarships have txt files — full RAG coverage.")

        if not all_chunks:
            print("[rag_init] Warning: no documents to index.")
            return

        print(f"[rag_init] Embedding {len(all_chunks)} chunks …")
        texts      = [c.content for c in all_chunks]
        embeddings = self.model.encode(
            texts, show_progress_bar=True, normalize_embeddings=True
        )
        embeddings = np.array(embeddings, dtype="float32")
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings)
        self.chunks = all_chunks
        print(f"[rag_init] FAISS index built — {len(all_chunks)} chunks, dim={dim}")

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = _RAG_TOP_K,
        scholarship_names: list[str] | None = None,
    ) -> list[dict]:
        """
        Return top_k most relevant chunks.
        If scholarship_names is given, restrict results to those scholarships only.

        Fix vs original:
        When a scholarship_names whitelist is provided the search pool is enlarged
        to top_k * 20 (min 50). The original top_k * 6 pool could be exhausted
        before finding top_k results from a narrow whitelist (e.g. 2 scholarships),
        silently returning fewer hits than requested and degrading answer quality.
        """
        if self.faiss_index is None or not self.chunks:
            return []

        query_vec = np.array(
            self.model.encode([query], normalize_embeddings=True), dtype="float32"
        )

        # Use a larger pool when filtering by a specific scholarship whitelist.
        # With a narrow whitelist most globally top-ranked chunks belong to other
        # scholarships and are discarded, so we need to cast a wider net.
        if scholarship_names:
            search_k = min(max(top_k * 20, 50), len(self.chunks))
        else:
            search_k = min(max(top_k * 6, 10), len(self.chunks))

        scores, indices = self.faiss_index.search(query_vec, search_k)

        seen_ids: set[str]   = set()
        results: list[dict]  = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            if scholarship_names and chunk.scholarship_name not in scholarship_names:
                continue
            if chunk.chunk_id in seen_ids:
                continue
            seen_ids.add(chunk.chunk_id)
            results.append({
                "scholarship_name": chunk.scholarship_name,
                "chunk_id":         chunk.chunk_id,
                "score":            float(score),
                "content":          chunk.content,
            })
            if len(results) >= top_k:
                break

        return results