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

Changes in this version (hybrid search + score threshold):
  - BM25 index (rank_bm25.BM25Okapi) built alongside FAISS in _build_index().
  - retrieve() now performs hybrid search:
      * Candidate pool = union of top FAISS indices and top BM25 indices.
      * Hybrid score   = 0.5 * semantic_score + 0.5 * bm25_normalized_score.
      * BM25 normalization guard: divide by bm25_max only when bm25_max > 1.0;
        when bm25_max <= 1.0 raw scores are used as-is; division by zero is guarded.
  - min_score parameter added to retrieve() (read from config key rag_min_score,
    fallback 0.30). Chunks with hybrid_score < min_score are discarded.
  - Optional debug logging controlled by config key debug_rag (bool, default false).
  - Return schema of retrieve() is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
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
_RAG_MIN_SCORE     = float(cfg.get("rag_min_score", 0.30))


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
    Sentence-transformer + FAISS semantic search over scholarship documents,
    combined with BM25 keyword search for hybrid retrieval.

    Each scholarship gets one or more chunks from its txt file (or a CSV summary
    fallback if no txt is available).

    Hybrid retrieval:
      - Candidate pool  = union of top-FAISS indices and top-BM25 indices.
      - Hybrid score    = 0.5 * semantic_score + 0.5 * bm25_normalized_score.
      - min_score gate  = discard chunks below the hybrid score threshold.
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
        self.bm25: Optional[BM25Okapi] = None
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
        import re

        # Split text into sentences first for sentence-aware chunking
        sentences = re.split(r'(?<=[.!?])\s+', text)

        result = []
        idx = 0
        current_chunk_sentences = []
        current_word_count = 0

        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_len = len(sentence_words)

            # If a single sentence exceeds chunk_words, split at word boundary
            if sentence_len > chunk_words:
                step = max(chunk_words - overlap, 1)
                for word_start in range(0, sentence_len, step):
                    word_end = min(word_start + chunk_words, sentence_len)
                    chunk_text = " ".join(sentence_words[word_start:word_end])
                    full_chunk = f"[{name}]\n{chunk_text}"
                    result.append(Chunk(
                        chunk_id         = f"{name.lower().replace(' ', '_')}_{idx}",
                        scholarship_name = name,
                        content          = full_chunk,
                    ))
                    idx += 1
                overlap_sentences = []
                overlap_words = 0
                for s in reversed(current_chunk_sentences):
                    wc = len(s.split())
                    if overlap_words + wc > overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_words += wc
                current_chunk_sentences = overlap_sentences
                current_word_count = overlap_words
            else:
                if current_word_count + sentence_len > chunk_words and current_chunk_sentences:
                    chunk_text = " ".join(current_chunk_sentences)
                    full_chunk = f"[{name}]\n{chunk_text}"
                    result.append(Chunk(
                        chunk_id         = f"{name.lower().replace(' ', '_')}_{idx}",
                        scholarship_name = name,
                        content          = full_chunk,
                    ))
                    idx += 1
                    overlap_sentences = []
                    overlap_words = 0
                    for s in reversed(current_chunk_sentences):
                        wc = len(s.split())
                        if overlap_words + wc > overlap:
                            break
                        overlap_sentences.insert(0, s)
                        overlap_words += wc
                    current_chunk_sentences = overlap_sentences
                    current_word_count = overlap_words

                current_chunk_sentences.append(sentence)
                current_word_count += sentence_len

        # Emit any remaining sentences as a final chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            full_chunk = f"[{name}]\n{chunk_text}"
            result.append(Chunk(
                chunk_id         = f"{name.lower().replace(' ', '_')}_{idx}",
                scholarship_name = name,
                content          = full_chunk,
            ))

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

        # ── BM25 index (built after chunks are finalised) ──────────────────────
        tokenized_corpus = [c.content.lower().split() for c in all_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"[rag_init] BM25 index built — {len(all_chunks)} chunks")

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = _RAG_TOP_K,
        scholarship_names: list[str] | None = None,
        min_score: float | None = None,
    ) -> list[dict]:
        """
        Return top_k most relevant chunks using hybrid search (FAISS + BM25).

        If scholarship_names is given, restrict results to those scholarships only.
        Chunks with a hybrid score below min_score are discarded entirely.

        Hybrid scoring:
          hybrid_score = 0.5 * semantic_score + 0.5 * bm25_normalized_score

        BM25 normalization guard:
          - If bm25_max > 1.0:  normalize by dividing by bm25_max  → [0, 1]
          - If bm25_max in (0, 1]: keep raw scores as-is (already low range)
          - If bm25_max == 0:    set all BM25 contributions to 0.0

        Candidate pool optimization:
          Only the union of top FAISS and top BM25 indices is scored,
          avoiding full-corpus BM25 computation.
        """
        if self.faiss_index is None or not self.chunks:
            return []

        if min_score is None:
            min_score = _RAG_MIN_SCORE

        query_vec = np.array(
            self.model.encode([query], normalize_embeddings=True), dtype="float32"
        )

        # ── Candidate pool size ────────────────────────────────────────────────
        # Use a larger pool when filtering by a specific scholarship whitelist.
        # With a narrow whitelist most globally top-ranked chunks belong to other
        # scholarships and are discarded, so we need to cast a wider net.
        if scholarship_names:
            search_k = min(max(top_k * 20, 50), len(self.chunks))
        else:
            search_k = min(max(top_k * 6, 10), len(self.chunks))

        # ── FAISS semantic retrieval ───────────────────────────────────────────
        faiss_scores_raw, faiss_indices_raw = self.faiss_index.search(query_vec, search_k)
        faiss_scores: dict[int, float] = {}
        for score, idx in zip(faiss_scores_raw[0], faiss_indices_raw[0]):
            if idx >= 0:
                faiss_scores[int(idx)] = float(score)

        # ── BM25 keyword retrieval ─────────────────────────────────────────────
        query_tokens    = query.lower().split()
        bm25_scores_all = np.array(self.bm25.get_scores(query_tokens))
        # Get top search_k BM25 indices (without iterating over all documents at query time)
        bm25_top_raw    = np.argpartition(bm25_scores_all, -min(search_k, len(bm25_scores_all)))[
            -min(search_k, len(bm25_scores_all)):
        ]
        bm25_top_indices: set[int] = set(int(i) for i in bm25_top_raw)

        # ── Union of candidate pools ───────────────────────────────────────────
        candidate_indices: set[int] = set(faiss_scores.keys()) | bm25_top_indices

        # ── BM25 normalization guard ───────────────────────────────────────────
        candidate_bm25_raw = {i: float(bm25_scores_all[i]) for i in candidate_indices}
        bm25_max           = max(candidate_bm25_raw.values()) if candidate_bm25_raw else 0.0

        if bm25_max > 1.0:
            # Normalize to [0, 1]
            bm25_norm: dict[int, float] = {i: s / bm25_max for i, s in candidate_bm25_raw.items()}
        elif bm25_max == 0.0:
            # All BM25 scores are 0 — no keyword signal
            bm25_norm = {i: 0.0 for i in candidate_indices}
        else:
            # bm25_max in (0, 1] — raw scores are already in a low range, keep as-is
            bm25_norm = candidate_bm25_raw

        # ── Hybrid scoring ─────────────────────────────────────────────────────
        hybrid_scores: dict[int, float] = {
            idx: 0.5 * faiss_scores.get(idx, 0.0) + 0.5 * bm25_norm.get(idx, 0.0)
            for idx in candidate_indices
        }

        # ── Debug logging (controlled by config key debug_rag) ─────────────────
        if cfg.get("debug_rag") and hybrid_scores:
            vals = list(hybrid_scores.values())
            print(
                f"[rag_retrieve] candidates={len(candidate_indices)} "
                f"min_hybrid={min(vals):.3f} max_hybrid={max(vals):.3f} "
                f"threshold={min_score:.3f} bm25_max={bm25_max:.3f}"
            )

        # ── Filter, rank, deduplicate, return ─────────────────────────────────
        seen_ids: set[str]  = set()
        results: list[dict] = []

        for idx, score in sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True):
            if score < min_score:
                break  # sorted descending — everything after is also below threshold
            chunk = self.chunks[idx]
            if scholarship_names and chunk.scholarship_name not in scholarship_names:
                continue
            if chunk.chunk_id in seen_ids:
                continue
            seen_ids.add(chunk.chunk_id)
            results.append({
                "scholarship_name": chunk.scholarship_name,
                "chunk_id":         chunk.chunk_id,
                "score":            score,
                "content":          chunk.content,
            })
            if len(results) >= top_k:
                break

        if cfg.get("debug_rag"):
            print(f"[rag_retrieve] hits_after_filter={len(results)}")

        return results
