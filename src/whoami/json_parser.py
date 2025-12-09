class JsonParser:
    PLACEHOLDERS = {
            "",
            "n/a",
            "na",
            "none",
            "unknown",
            "?",
            "no character",
            "human character",
            "fictional character",
            "real person",
            "human",
            "person",
    }
    
    @classmethod
    def parse_llm_output(cls, data) -> tuple[str, list, list]:
        """
        Parse the JSON from the main game LLM into:
        - robot utterance
        - profile (character facts)
        - most_likely (cleaned candidates)
        """
        if not isinstance(data, dict):
            return str(data), [], []

        response_text = data.get("response") or ""
        profile = data.get("profile") or []
        raw_candidates = data.get("most_likely") or []

        cleaned_candidates = []
    
        if isinstance(raw_candidates, list):
            for item in raw_candidates[:3]:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                why = str(item.get("why", "")).strip()
                try:
                    likelihood = float(item.get("likelihood", 0) or 0)
                except Exception:
                    likelihood = 0.0

                if name.lower() in cls.PLACEHOLDERS:
                    continue

                cleaned_candidates.append(
                    {"name": name, "why": why, "likelihood": likelihood}
                )

        # Bug fix where all candidates have likelihoods of zero
        if cleaned_candidates:
            all_zero = all((c.get("likelihood", 0.0) == 0.0) for c in cleaned_candidates)
            if all_zero:
                n = len(cleaned_candidates)
                if n > 0:
                    equal_prob = round(1.0 / n, 2)
                    for c in cleaned_candidates:
                        c["likelihood"] = equal_prob

        return response_text, profile, cleaned_candidates