"""
llm_handler.py
──────────────
Thin wrapper around the local Ollama API.

Changes vs original:
  - Added `use_json_format` parameter to chat().
    When True, passes format="json" to Ollama, forcing the model to emit
    structurally valid JSON at the inference level — not just instructed to.
    This eliminates the largest class of router failures (preamble text,
    markdown fences, mid-generation drift) without any prompt changes.
  - `do_sample` parameter removed. At temperature=0.0 sampling is
    deterministic regardless; at temperature>0 Ollama handles it correctly
    without an explicit flag. Passing do_sample had no effect on the Ollama
    backend and was confusing.
"""

import json
import re

import ollama

from config_loader import cfg

_OLLAMA_MODEL = cfg["ollama_model"]


class LocalLLM:
    def __init__(self, model_name: str = _OLLAMA_MODEL):
        self.model_name = model_name
        print(f"[llm_init] Using Ollama model: {self.model_name}")
        # Warm-up ping so any model-load delay happens at startup, not mid-chat
        try:
            ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": "hi"}],
                options={"num_predict": 1},
            )
            print("[llm_init] Ollama connection verified — model ready")
        except Exception as e:
            print(f"[llm_init] Warning: Ollama warm-up failed: {e}")

    def chat(
        self,
        messages: list[dict],
        max_new_tokens: int = 768,
        temperature: float = 0.0,
        use_json_format: bool = False,
    ) -> str:
        """
        Call the model.

        Parameters
        ----------
        messages         : standard OpenAI-style message list
        max_new_tokens   : token budget for the response
        temperature      : 0.0 = deterministic (use for router + answer LLM);
                           0.1 = near-deterministic with slight variation
        use_json_format  : if True, passes format="json" to Ollama so the model
                           is constrained to produce syntactically valid JSON.
                           Always use this for router/extractor calls.
        """
        try:
            call_kwargs: dict = {
                "model":    self.model_name,
                "messages": messages,
                "options":  {
                    "num_predict": max_new_tokens,
                    "temperature": temperature,
                },
            }
            if use_json_format:
                # Ollama's format="json" constrains token generation to valid JSON.
                # This is not the same as instructing the model to output JSON —
                # it's a hard inference-level constraint that eliminates most
                # malformed-output failures before they reach _extract_json().
                call_kwargs["format"] = "json"

            response = ollama.chat(**call_kwargs)
            return response["message"]["content"].strip()
        except Exception as e:
            return f"⚠️ Generation error: {e}"

    def extract_json(self, text: str) -> dict | None:
        """Try several strategies to extract a JSON object from raw LLM output."""
        # 1. Direct parse
        try:
            return json.loads(text)
        except Exception:
            pass

        # 2. Strip markdown fences then parse
        cleaned = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        # 3. Extract first {...} block
        match = re.search(r"\{[\s\S]+\}", cleaned)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass

        # 4. Light repairs (Python literals → JSON) then try again
        repaired = re.sub(r",\s*([}\]])", r"\1", cleaned)  # trailing commas
        repaired = re.sub(r"\bTrue\b",  "true",  repaired)
        repaired = re.sub(r"\bFalse\b", "false", repaired)
        repaired = re.sub(r"\bNone\b",  "null",  repaired)
        try:
            return json.loads(repaired)
        except Exception:
            pass

        return None