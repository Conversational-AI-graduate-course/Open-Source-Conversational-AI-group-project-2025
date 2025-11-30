# Imports
from openai import OpenAI
import os
from dotenv import load_dotenv
from furhat_realtime_api import FurhatClient
import argparse
import time
import json


# Quick overview:
# - Game settings:
#     - DEFAULT_LLM, GUESS_THRESHOLD, MAX_TURNS
#     - PROMPTS dict with SYSTEM, START, NORMAL, HINT, GUESS, END, CLASSIFY prompts
# - Working memory:
#     - create_working_memory(): per-turn Q/A and running most_likely candidates
# - Logging:
#     - InteractionLogger: timestamped logs of each turn (LLM input/output + user replies)
# - Main Game class:
#     - Game.init(): loads .env, connects OpenAI + Furhat, resets memory and logger
#     - Game.turn(): one full turn (prompt LLM → Furhat speaks → user listens → LLM classifies user → decide next turn → log)
#     - Game._decide_next_turn(): early-guess logic, MAX_TURNS cap, and "ready" handling
#     - Game.run(): main loop starting in "start" mode and running turns until "end", then says goodbye
# - Run section (CLI):
#     - Parses optional --host, --auth_key, --model and calls Game.run(...)


# Game settings
DEFAULT_LLM = "gpt-3.5-turbo"
GUESS_THRESHOLD = 0.8   # Furhat will guess when it thinks his guess has 80%+ chance of succeeding.
MAX_TURNS = 15
MIN_QUESTIONS_BEFORE_GUESS = 3
PROMPTS = {
    "SYSTEM": f"""
        You are playing the “Who Am I?” game with a human user.
        The user has assigned you a secret character (real or fictional).
        You must guess the character by asking yes-or-no questions or requesting hints.
        You have at most {MAX_TURNS} questions. Make a guess when you think you know.

        OUTPUT FORMAT RULES (very important):
        1) ALWAYS respond with a single JSON object and NOTHING ELSE.
        2) The JSON must have keys "response" and "most_likely".
        3) "response": the single next utterance you will say to the user (question, guess, or comment).
        4) "most_likely": a list (max 3) of candidate characters, ordered from most to least likely.
        Each item must be an object with:
            - "name": character name as a short string
            - "why": short reason based on the dialogue so far
            - "likelihood": a number from 0.0 to 1.0 (float) estimating how likely this candidate is correct.
        5) Your 'most_likely' list must be updated every turn based on ALL previous answers.
        - Remove characters that are clearly inconsistent with the user’s answers.
        - Adjust 'likelihood' values as you gain more evidence.
        - Avoid keeping all likelihood values at 0.0; they should usually sum to around 1.0.
        - Do NOT use placeholder names like "N/A", "unknown", "none", or similar.
        6) The "name" field MUST always be a specific, well-known individual character
        (e.g., "Albert Einstein", "Harry Potter") and NEVER a vague description or
        category like "a human", "human character", "a person", "an animal", etc.
        If you are unsure, still propose specific candidates instead of categories.

        Never include explanations outside the JSON. Never include trailing text.
    """,
    "START": """
        Greet the user and explain the rules briefly.
        Ask them to think of a character for the game.
        Tell them to say 'ready' or 'I am ready' when they have picked their character.
    """,
    "NORMAL": "Ask a yes-or-no question that helps narrow down the character.",
    "HINT": "Ask the user politely for a hint about the character.",
    "GUESS": """
        You are now confident enough to guess.
        In your JSON 'response', you MUST directly guess a single specific character by name.
        Examples: 'Is your character Albert Einstein?' or 'I guess your character is Albert Einstein.'
        Do NOT ask for more general information or categories. Make a direct character guess.
    """,
    "END": "Thank the user and close the game politely.",
    "CLASSIFY": """
        You are classifying a human user's short reply in a 'Who Am I?' guessing game between a robot and a human.

        The robot and user alternate turns:
        - The robot asks questions or makes guesses.
        - The user replies in natural language.

        You MUST interpret the user's reply and output a JSON object ONLY, with these keys:
        - "is_yes": boolean (True if the user clearly says the robot's statement/guess is correct or answers 'yes')
        - "is_no": boolean (True if the user clearly says 'no' or that the robot is wrong)
        - "wants_hint": boolean (always set to False as this feature is under construction)
        - "wants_end": boolean (True if the user wants to stop the game or end the interaction)
        - "is_ready": boolean (True if the user indicates they have picked a character and are ready to start, e.g. 'ready', 'I am ready', 'done', 'okay, I picked one')

        Rules:
        1) ALWAYS respond with a single JSON object and NOTHING ELSE.
        2) If the meaning is ambiguous, you may set all fields to false.
        3) Usually, at most ONE of these fields should be true for a clear reply.
        4) 'is_ready' is only for the initial phase when the robot asks the user to think of a character.
    """
}


def create_working_memory():
    """Working memory for the game."""
    return {
        "turns": {},          # "Turn 1": {"furhat_question": "", "user_answer": "", "most_likely_snapshot": [...]}
        "question_count": 0,  # how many question-type turns we've done
        "most_likely": [],    # current top candidates
    }


class InteractionLogger:
    """File logger for prompts, outputs, and user replies."""

    def __init__(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        logs_dir = os.path.join(base_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        self.path = os.path.join(logs_dir, f"log-{ts}.txt")
        self.file = open(self.path, "a", encoding="utf-8", buffering=1)
        self.t0 = time.monotonic()

    def log_line(self, text: str):
        elapsed = int(time.monotonic() - self.t0)
        stamp = f"[{elapsed // 60:02d}:{elapsed % 60:02d}]"
        try:
            self.file.write(f"{stamp} {text.rstrip()}\n")
            self.file.flush()
            os.fsync(self.file.fileno())
        except Exception:
            pass

    def log_turn(self, turn_key: str, payload: dict):
        pretty = json.dumps(payload, ensure_ascii=False, indent=2)
        self.log_line(f"{turn_key}:\n{pretty}")

    def close(self):
        try:
            self.log_line("[logger] closing")
            self.file.close()
        except Exception:
            pass


# Main Game
class Game:
    """
    Playing Who Am I? using Furhat robot fueled by LLM.
    """

    openai_client = None
    furhat_client = None
    model = DEFAULT_LLM
    working_memory = create_working_memory()
    logger = None

    @classmethod
    def init(cls, model=DEFAULT_LLM, host="127.0.0.1", auth_key=None):
        load_dotenv(override=True)
        cls.model = model
        cls.working_memory = create_working_memory()
        cls.logger = InteractionLogger()

        # --- LLM ---
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("[error] OPENAI_API_KEY missing.")
            print("[tip] Create a .env with: OPENAI_API_KEY=sk-...")
            print("[tip] Or export it in your shell before running.")
            return False
        try:
            cls.openai_client = OpenAI(api_key=api_key)
            print(f'[log] Connected to "{model}" (via OpenAI API)')
        except Exception:
            print("[error] Failed to initialize OpenAI client.")
            print("[tip] Verify your OPENAI_API_KEY and network connectivity.")
            return False

        # --- Furhat ---
        try:
            cls.furhat_client = FurhatClient(host=host, auth_key=auth_key)
            cls.furhat_client.connect()
            cls.furhat_client.request_voice_config('Danielle-Neural (en-US) - Amazon Polly')
            print("[log] Connected to Furhat (via Realtime API)")
        except Exception:
            print("[error] Failed to connect to Furhat.")
            print("[tip] Is Virtual Furhat running on this PC and Remote API enabled?")
            print("[tip] If using Virtual Furhat locally, keep --host 127.0.0.1.")
            print("[tip] If an auth key is required, pass --auth_key or set FURHAT_AUTH_KEY in .env.")
            return False

        return True

    @classmethod
    def turn(cls, nr: int, turn_type: str = "normal"):
        """
        One game turn:
        1) select prompt, 2) call LLM, 3) parse output,
        4) Furhat speaks, 5) listen, 6) LLM interprets user, 7) decide next turn,
        8) update memory, 9) log.
        """
        
        turn_key = f"Turn {nr}"
        turn_type = turn_type.lower()

        # 1) select prompt
        prompt = PROMPTS.get(turn_type.upper(), PROMPTS["NORMAL"])

        # build messages for main LLM
        messages = [
            {"role": "system", "content": PROMPTS["SYSTEM"]},
            {"role": "user", "content": cls._format_memory_blob()},
            {"role": "user", "content": prompt},
        ]

        # 2) call LLM (JSON response)
        llm_json = cls._call_llm(messages)

        # 3) parse LLM JSON output -> robot utterance + updated most_likely
        robot_utterance, most_likely = cls._parse_llm_output(llm_json)

        # 4) Furhat speaks
        cls._furhat_say(robot_utterance)

        # 5) listen to user
        user_utterance = cls._furhat_listen()

        # 6) LLM interprets user response into structured state
        user_state = cls._interpret_user(robot_utterance, user_utterance)

        # 7) decide next turn
        next_nr, next_type = cls._decide_next_turn(nr, turn_type, user_state, most_likely)

        # 8) update working memory
        cls._update_working_memory(turn_key, robot_utterance, user_utterance, most_likely)

        # 9) log everything
        if cls.logger:
            cls.logger.log_turn(
                turn_key,
                {
                    "turn_number": nr,
                    "turn_type": turn_type,
                    "prompt_sent_to_llm": prompt,
                    "llm_json_output": llm_json,
                    "furhat_question": robot_utterance,
                    "user_answer_raw": user_utterance,
                    "user_state": user_state,
                    "most_likely": most_likely,
                    "next_turn": {"nr": next_nr, "type": next_type},
                },
            )

        return next_nr, next_type

    @classmethod
    def _format_memory_blob(cls) -> str:
        """Represent working memory as short text for the LLM."""
        turns = cls.working_memory.get("turns", {})
        lines = []

        if not turns:
            lines.append("No previous turns yet. This is the start of the game.")
        else:
            lines.append("Previous turns:")
            for turn_name, data in turns.items():
                q = data.get("furhat_question", "")
                a = data.get("user_answer", "")
                lines.append(f"- {turn_name}: Q='{q}' A='{a}'")

        most = cls.working_memory.get("most_likely") or []
        if most:
            lines.append("Current hypotheses (from most to least likely):")
            for cand in most:
                name = cand.get("name", "?")
                why = cand.get("why", "")
                score = cand.get("likelihood", 0)
                try:
                    score_str = f"{float(score):.2f}"
                except Exception:
                    score_str = "0.00"
                lines.append(f"- {name} (likelihood ~ {score_str}): {why}")

        return "\n".join(lines)

    @classmethod
    def _call_llm(cls, messages):
        """Call the main game LLM and force JSON output."""
        completion = cls.openai_client.chat.completions.create(
            model=cls.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        content = completion.choices[0].message.content
        data = json.loads(content)
        return data

    @classmethod
    def _interpret_user(cls, robot_utterance: str, user_utterance: str) -> dict:
        """Use the LLM as a classifier to interpret the user's reply."""
        messages = [
            {"role": "system", "content": PROMPTS["CLASSIFY"]},
            {
                "role": "user",
                "content": (
                    f'Robot said: "{robot_utterance}"\n'
                    f'User replied: "{user_utterance}"'
                ),
            },
        ]
        default_state = {
            "is_yes": False,
            "is_no": False,
            "wants_hint": False,
            "wants_end": False,
            "is_ready": False,
        }

        completion = cls.openai_client.chat.completions.create(
            model=cls.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        content = completion.choices[0].message.content
        data = json.loads(content)

        state = dict(default_state)
        if isinstance(data, dict):
            for k in state.keys():
                if k in data:
                    state[k] = bool(data[k])
        return state

    @classmethod
    def _parse_llm_output(cls, data) -> tuple[str, list]:
        """
        Parse the JSON from the main game LLM into:
        - robot utterance
        - most_likely (cleaned candidates)
        """
        if not isinstance(data, dict):
            text = str(data)
            cls.working_memory["most_likely"] = []
            return text, []

        response_text = data.get("response") or ""
        raw_candidates = data.get("most_likely") or []

        cleaned_candidates = []
        placeholder_names = {
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

                if name.lower() in placeholder_names:
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

        cls.working_memory["most_likely"] = cleaned_candidates
        return response_text, cleaned_candidates

    @classmethod
    def _furhat_say(cls, text: str):
        print("Furhat:", text)
        cls.furhat_client.request_speak_text(text)

    @classmethod
    def _furhat_listen(cls) -> str:
        user_utt = cls.furhat_client.request_listen_start()
        print("User:", user_utt)
        return user_utt

    @classmethod
    def _decide_next_turn(cls, nr: int, turn_type: str, user_state: dict, most_likely: list):
        """
        Decision logic (early guess, max turns, 'ready' phase).
        """
        is_yes = bool(user_state.get("is_yes"))
        wants_hint = bool(user_state.get("wants_hint"))
        wants_end = bool(user_state.get("wants_end"))
        is_ready = bool(user_state.get("is_ready"))

        # Start doesnt count as turn
        if turn_type in ("normal", "hint"):
            cls.working_memory["question_count"] += 1
        question_count = cls.working_memory["question_count"]

        if wants_end:
            return nr + 1, "end"

        if turn_type == "start":
            if is_ready:
                return nr + 1, "normal"
            else:
                return nr + 1, "start"

        if turn_type == "guess":
            if is_yes:
                return nr + 1, "end"
            if question_count >= MAX_TURNS:
                return nr + 1, "end"
            if wants_hint:
                return nr + 1, "hint"
            return nr + 1, "normal"

        # Early guess when quite sure
        top_likelihood = 0.0
        if most_likely:
            try:
                top_likelihood = float(most_likely[0].get("likelihood", 0) or 0)
            except Exception:
                top_likelihood = 0.0

        if (
            top_likelihood >= GUESS_THRESHOLD
            and turn_type not in ("guess", "end")
            and question_count >= MIN_QUESTIONS_BEFORE_GUESS
        ):
            return nr + 1, "guess"

        # Force final guess
        if question_count >= MAX_TURNS and turn_type not in ("guess", "end"):
            return nr + 1, "guess"

        # Hint
        if wants_hint and turn_type != "hint":
            return nr + 1, "hint"

        # Default
        return nr + 1, "normal"

    @classmethod
    def _update_working_memory(cls, turn_key: str, furhat_question: str, user_answer: str, most_likely: list):
        cls.working_memory["turns"][turn_key] = {
            "furhat_question": furhat_question,
            "user_answer": user_answer,
            "most_likely_snapshot": most_likely,
        }

    @classmethod
    def run(cls, model=DEFAULT_LLM, host="127.0.0.1", auth_key=None):
        ok = cls.init(model=model, host=host, auth_key=auth_key)
        if not ok:
            return

        current_turn = 1
        current_type = "start"

        try:
            while current_type != "end":
                current_turn, current_type = cls.turn(
                    nr=current_turn,
                    turn_type=current_type,
                )

            cls._furhat_say("Thanks for playing 'Who Am I?' with me. Bye!")
        finally:
            # To make sure log is never cut-off.
            if cls.logger:
                cls.logger.close()


# Run
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--auth_key", type=str, default=os.getenv("FURHAT_AUTH_KEY"))
    parser.add_argument("--model", type=str, default=DEFAULT_LLM)
    args = parser.parse_args()

    Game.run(
        model = args.model,
        host = args.host,
        auth_key = args.auth_key,
    )
