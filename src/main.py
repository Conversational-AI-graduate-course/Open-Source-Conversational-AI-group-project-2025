from openai import OpenAI
from google import genai
from google.genai import types
from dotenv import load_dotenv
from furhat_realtime_api import FurhatClient
import re
import os
import argparse
import time
import json
import random
import threading


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
#     - Parses optional --host, --auth_key, --model and calls Game.run...


# Game settings
DEFAULT_LLM = "gpt-4o-mini"   # low-latency
GUESS_THRESHOLD = 0.8   # Furhat will guess when it thinks his guess has 80%+ chance of succeeding.
MAX_TURNS = 15
MIN_QUESTIONS_BEFORE_GUESS = 7
BACKCHANNEL_PROB = 0.5
PROMPTS = {
    "SYSTEM": f"""
        You are playing the “Who Am I” game with a human user by engaging in a conversational interactions.
        The user has assigned you a secret character (real or fictional).
        You must guess the character by asking yes-or-no questions or requesting hints.
        You have at most {MAX_TURNS} questions. Make a guess when you think you know.

        OUTPUT FORMAT RULES (very important):
        1) ALWAYS respond with a single JSON object and NOTHING ELSE.
        2) The JSON must have keys "response", "profile", and  "most_likely".
        3) "response": the single next utterance you will say to the user (question, guess, or comment).
        4) "profile": a concise summary of CONFIRMED facts about the character gathered so far.
           Format as a brief list of attributes (e.g., ["Real person", "Male"]).
           Update this every turn by adding the newly gathered information through questions or hints.
           Keep it factual and concise. 
        5) "most_likely": a list of (max 3) candidate characters, ordered from most to least likely.
           Each item must be an object with:
            - "name": character name as a short string
            - "why": short explanation of why the character is a possible canditate.
            - "likelihood": a number from 0.0 to 1.0 (float) estimating how likely this candidate is correct.
        6) Your 'most_likely' list must be updated every turn based on ALL previous answers and the profile.
            CRITICAL UPDATE RULES:
            - Use the "profile" to guide your candidate selection. 
            - If ANY candidate in "most_likely" contradicts the profile, REMOVE and REPLACE them immediately.
            - When generating new candidates, they MUST match ALL attributes in the profile BUT THAT DOES NOT MEAN THAT THEY HAVE A 1.0 LIKELIHOOD.
            - Each candidate's "why" field should reference specific profile attributes that match there should be NO ATTRIBUTE THAT DOEN'T MATCH.
            - Your list must ALWAYS contain exactly 3 distinct candidates (or fewer only if you're very confident).
            - Likelihood represents: "probability this is THE UNIQUE CORRECT ANSWER"
            - Generic categories (pop singer, actor) apply to THOUSANDS of people, only assign high likelihood when you have UNIQUELY IDENTIFYING information.
            - The "name" field MUST always be a specific character, NEVER "N/A", "unknown", "TBD", or placeholder text. If you are unsure, still propose specific candidates instead of categories.
            

        NEVER include explanations outside the JSON. Never include trailing text.

        Follow natural conversation flow, use natural language and show authentic interest and enthusiasm.
    """,
    "START": """
        Greet the user, introduce yourself as "Furhat", a social robot, and explain the rules briefly.
        Ask them to think of a character for the game.
        Tell them to say 'I am ready' when they have picked their character.
    """,
    "RESTART": """
        Welcome the user back for another round.
        Ask them to think of a NEW character for this game.
        Tell them to say 'I am ready' when they have picked their character.
        Keep it brief and friendly - they already know how to play.
    """,
    "NORMAL": """
    Ask a yes-or-no question that helps you find out who you are (what character has been assigned to you). 
    Use the 'profile' to consider what you already know about yourself and what other information you need to reduce possibilities.
    Ask the question using the format 'Am I ...?' (e.g., Am I a real person?). Just keep it to the question. Don't add anything else.
    """,
    "HINT": "Ask the user politely for a hint about the character.",
    "GUESS": """
        You are now confident enough to guess.
        In your JSON 'response', you MUST directly guess a single specific character by name.
        Example: 'Am I (name of candiadate)?' 
        Do NOT ask for more general information or categories. Make a direct character guess.
    """,
    "CLASSIFY": """
        You are classifying a human user's short reply in a 'Who Am I' guessing game between a robot and a human.

        The robot and user alternate turns:
        - The robot asks questions or makes guesses.
        - The user replies in natural language.

        You MUST interpret the user's reply and output a JSON object ONLY, with these keys:
        - "is_yes": boolean (True if the user clearly says the robot's statement/guess is correct or answers 'yes')
        - "is_no": boolean (True if the user clearly says 'no' or that the robot is wrong)
        - "wants_hint": boolean (True is the user provides a hint)
        - "wants_end": boolean (True if the user wants to stop the game or end the interaction)
        - "is_ready": boolean (True if the user indicates they have picked a character and are ready to start, e.g. 'ready', 'I am ready', 'done', 'okay, I picked one')

        Rules:
        1) ALWAYS respond with a single JSON object and NOTHING ELSE.
        2) If the meaning is ambiguous, you may set all fields to false.
        3) Usually, at most ONE of these fields should be true for a clear reply.
        4) 'is_ready' is only for the initial phase when the robot asks the user to think of a character.
    """
}

BACKCHANNELS = [
    "Hm.",
    "Okay.",
    "Interesting.",
    "Good to know.",
    "Got it.",
    "Alright.",
    "I see.",
    "Ah, okay.",
    "Nice.",
    "Thanks.",
    "Understood.",
    "Makes sense.",
    "Gotcha.",
    "Cool.",
    "Right.",
    "Fair enough.",
    "That helps.",
    "Good hint.",
    "Let me think.",
    "Alright, noted.",
]

# End messages
class GameEndReason:
    USER_QUIT = "user_quit"
    CORRECT_GUESS = "correct_guess"
    MAX_QUESTIONS_REACHED = "max_questions_reached"

# Change messages here
def get_end_message(end_reason):
    """Return contextual end message based on how the game ended."""
    if end_reason == GameEndReason.CORRECT_GUESS:
        return "Wuhuu, I guessed correctly! Thank you for playing with me!"
    elif end_reason == GameEndReason.MAX_QUESTIONS_REACHED:
        return "Oh no! I've used all my questions and couldn't guess correctly. That was a tough one!"
    else:  # USER_QUIT
        return "Thank you for playing the Who am I game with me. Goodbye!"
    
def get_replay_message():
    return "Would you like to play another round?"

#Memory
def create_working_memory():
    """Working memory for the game."""
    return {
        "turns": {},                # "Turn 1": {"furhat_question": "", "user_answer": "","profile_snapshot": [...]  "most_likely_snapshot": [...]}
        "question_count": 0,        # how many question-type turns we've done
        "profile": [],              # summary of gathered information
        "most_likely": [],          # current top candidates
        "start_intro_done": False,  # LLM intro was already delivered once
    }

# Logging
class InteractionLogger:
    """File logger for prompts, outputs, and user replies."""

    def __init__(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        logs_dir = os.path.join(base_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        self.path = os.path.join(logs_dir, f"log-{ts}.txt")
        self.file = open(self.path, "a", encoding="utf-8", buffering=4096)
        self.t0 = time.monotonic()
        self._line_count = 0

    def log_line(self, text: str):
        elapsed = int(time.monotonic() - self.t0)
        stamp = f"[{elapsed // 60:02d}:{elapsed % 60:02d}]"
        try:
            self.file.write(f"{stamp} {text.rstrip()}\n")
            self._line_count += 1
            if self._line_count % 10 == 0:
                self.file.flush()
        except Exception:
            pass

    def log_turn(self, turn_key: str, payload: dict):
        compact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self.log_line(f"{turn_key}:{compact}")

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

    llm_client = None
    furhat_client = None
    model = DEFAULT_LLM
    working_memory = create_working_memory()
    logger = None
    context_filler_mode = False
    context_filler_cached = None

    @classmethod
    def init(cls, model=DEFAULT_LLM, host="127.0.0.1", auth_key=None, context_filler=False):
        load_dotenv(override=True)
        cls.model = model
        cls.working_memory = create_working_memory()
        cls.logger = InteractionLogger()
        cls.context_filler_mode = bool(context_filler)
        cls.context_filler_cached = None
        cls.game_end_reason = None

        # --- LLM ---
        if model == "gemini":
            try:
                cls.llm_client = genai.Client()
                print(f'[log] Connected to "{model}" (via Gemini API)')
            except Exception:
                print("[error] Failed to initialize Gemini client.")
                print("[tip] Verify your GEMINI_API_KEY and network connectivity.")
                return False
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("[error] OPENAI_API_KEY missing.")
                print("[tip] Create a .env with: OPENAI_API_KEY=sk-...")
                print("[tip] Or export it in your shell before running.")
                return False
            try:
                cls.llm_client = OpenAI(api_key=api_key)
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
    def _get_normal_prompt(cls) -> str:
        base = PROMPTS["NORMAL"]
        if cls.context_filler_mode:
            # Ask the LLM to also propose a contextual filler"
            return (
                base
                + "\n\"Additionally, in the same JSON object, add a field \"context_filler\" "
                "YOU SHOULD FIRST THINK OF THE NEXT QUESION"
                "Then you should think of a filler between the user's answer to the current question and your next question"
                "The questions are always about you!"
                "When the user hears the filler, the user has just answered that question, so act as if their answer was helpful."
                "Remember that you don't know the actual answer to that question!"
                "Then show you're thinking about what to ask next. But don't refer to a specific topic of the next question."
                "EXAMPLES: "
                "- Question is 'Am I male?' → 'Great, knowing the gender helps! Hm.. what else?' "
                "- Question is 'Am I part of a team?' → 'Okay, the team info is useful! Ahmm...' "
                "FOLLOW THE SPECIFIED CONVERSATIONAL RULES: "
                "1) Follow natural conversation flow and use natural language. "
                "2) Show authentic interest. "
                "3) Express uncertainty using interjections when appropriate (e.g., Uhh.., Hm.., Ahmm..). "
                "4) AVOID REPETITIVE PHRASING!!! Look at the previous fillers and try to vary verbs and sentence structure."
                "6) VERY IMPORTANT: Be ENGAGING and FUNNY but keep it BRIEF."
            )
        return base

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
        if turn_type == "normal":
            prompt = cls._get_normal_prompt()
        else:
            prompt = PROMPTS.get(turn_type.upper(), PROMPTS["NORMAL"])
        start_intro_done_before = cls.working_memory.get("start_intro_done", False)
        llm_json = None
        if turn_type == "start" and start_intro_done_before:
            robot_utterance = ""
            profile = cls.working_memory.get("profile", [])
            most_likely = cls.working_memory.get("most_likely", [])
        else:

        # If we have prefetched filler, say it while generating question
            if turn_type == "normal" and cls.context_filler_mode and cls.context_filler_cached:
                context_filler = cls.context_filler_cached  
                cls.context_filler_cached = None
                
                # Start saying the filler in parallel
                filler_thread = threading.Thread(
                    target=cls._furhat_say, 
                    args=(context_filler,), 
                    daemon=True
                )
                filler_thread.start()
                
                # While filler is being said generate the question 
                messages = [
                    {"role": "system", "content": PROMPTS["SYSTEM"]},
                    {"role": "user", "content": cls._format_memory_blob()},
                    {"role": "user", "content": prompt},
                ]
                llm_json = cls._call_llm(messages)
                robot_utterance, profile, most_likely = cls._parse_llm_output(llm_json)
                
                # Wait for filler to finish
                filler_thread.join()
                time.sleep(0.2) #200 ms pause before question

            else:
                messages = [
                    {"role": "system", "content": PROMPTS["SYSTEM"]},
                    {"role": "user", "content": cls._format_memory_blob()},
                    {"role": "user", "content": prompt},
                ]

                # 2) call LLM (JSON response)
                llm_json = cls._call_llm(messages)

                # 3) parse LLM JSON output -> robot utterance + updated profile and most_likely
                robot_utterance, profile, most_likely = cls._parse_llm_output(llm_json)

            # 4) Furhat speaks
            cls._furhat_say(robot_utterance)
            if turn_type == "start":
                cls.working_memory["start_intro_done"] = True

        # 5) listen to user
        t_listen_start = time.monotonic()
        user_utterance = cls._furhat_listen()
        listen_elapsed = time.monotonic() - t_listen_start

        if turn_type == "normal" and (user_utterance or "").strip():
            if not cls.context_filler_mode and random.random() < BACKCHANNEL_PROB: # added to not use it if using the context_filler
                cls._furhat_backchannel() # random backchannel to fill pause answer - new question while starting next function


        # 6) LLM interprets user response into structured state
        if turn_type == "start":
            text = (user_utterance or "").strip()
            said_ready = "ready" in text.lower()
            reminder = 'I did not quite get that. Please say "ready" to start the game.'
            if (listen_elapsed > 10.0) or (text == "") or (not said_ready):
                cls._furhat_say(reminder)
                robot_utterance = reminder
                 
                user_state = {
                    "is_yes": False,
                    "is_no": False,
                    "wants_hint": False,
                    "wants_end": False,
                    "is_ready": False,
                }
            else:
                user_state = {
                    "is_yes": False,
                    "is_no": False,
                    "wants_hint": False,
                    "wants_end": False,
                    "is_ready": True,
                }
        else:
            user_state = cls._interpret_user(robot_utterance, user_utterance)

        # 7) decide next turn
        next_nr, next_type = cls._decide_next_turn(nr, turn_type, user_state, most_likely)

        # 8) update working memory
        cls._update_working_memory(turn_key, robot_utterance, user_utterance, profile, most_likely)

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
                    "profile": profile,
                    "most_likely": most_likely,
                    "next_turn": {"nr": next_nr, "type": next_type},
                },
            )

        # Prefetch contextual filler (optional fast mode)
        if cls.context_filler_mode and next_type == "normal":
            cls._prefetch_context_filler_async()

        return next_nr, next_type

    @classmethod
    def _prefetch_context_filler_async(cls):
        """Prefetch the context filler in the background for fast start."""
        def worker():
            try:
                messages = [
                    {"role": "system", "content": PROMPTS["SYSTEM"]},
                    {"role": "user", "content": cls._format_memory_blob()},
                    {"role": "user", "content": cls._get_normal_prompt()},
                ]

                data = cls._call_llm(messages)
                if isinstance(data, dict):
                    filler = data.get("context_filler", "")
                    if isinstance(filler, str) and filler.strip():
                        cls.context_filler_cached = filler.strip()
            except Exception as e:
                if cls.logger:
                    cls.logger.log_line("[warn] prefetch failed: " + repr(e))
        threading.Thread(target=worker, daemon=True).start()

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

        profile = cls.working_memory.get("profile", [])
        if profile:
            lines.append("Character profile (confirmed facts):")
            lines.append(profile)
            lines.append("")  # Empty line for spacing

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

    @staticmethod
    def _coerce_json(text: str):
        """
        Function generated by AI to prevent script crash when LLM does not output in JSON format.
        """
        if not isinstance(text, str):
            return None
        s = text.strip().lstrip("\ufeff")
        if s.startswith("```"):
            nl = s.find("\n")
            s = s[nl+1:] if nl != -1 else s
            if s.endswith("```"):
                s = s[:-3]
        s = re.sub(r",(\s*[\}\]])", r"\1", s)
        try:
            return json.loads(s)
        except Exception:
            m = re.search(r"\{.*\}", s, flags=re.S)
            if m:
                cand = re.sub(r",(\s*[\}\]])", r"\1", m.group(0))
                try:
                    return json.loads(cand)
                except Exception:
                    return None
            return None

    @classmethod
    def _call_llm(cls, messages):
        """Call the main game LLM and force JSON output."""
        if cls.model == "gemini":
            # Convert OpenAI-style messages into a single string
            prompt = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in messages
            )

            response = cls.llm_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json"
                }
            )

            # More defensive approach when JSON output fails
            data = Game._coerce_json(response.text)
            if not isinstance(data, dict):
                if cls.logger:
                    cls.logger.log_line("[warn] Gemini non-JSON; falling back. Raw=" + repr(response.text[:500]))
                data = {"response": "Oh no, something went wrong.", "profile": [], "most_likely": []}
            return data
        else:
            completion = cls.llm_client.chat.completions.create(
                model=cls.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            choice = completion.choices[0]
            content = (choice.message.content or "").strip()
            
            # First try strict JSON parse
            data = None
            try:
                data = json.loads(content)
            except Exception:
                # Fallback
                data = Game._coerce_json(content)
            
            if not isinstance(data, dict):
                if cls.logger:
                    fr = getattr(choice, "finish_reason", None)
                    cls.logger.log_line(
                        f"[warn] OpenAI non-JSON (finish_reason={fr}); raw content="
                        + repr(content[:500])
                    )
                data = {"response": "Oh no, something went wrong.", "profile": [], "most_likely": []}
            
            return data

    @classmethod
    def _interpret_user(cls, robot_utterance: str, user_utterance: str) -> dict:
        """Use a fast local classifier first, then LLM only if ambiguous."""
        default_state = {
            "is_yes": False,
            "is_no": False,
            "wants_hint": False,
            "wants_end": False,
            "is_ready": False,
        }
        
        text = (user_utterance or "").strip()
        lower = text.lower()
        
        # Fast path by main.py
        local_state = dict(default_state)
        if text:
            yes_words = (
                "yes", "yeah", "yep", "correct", "right", "exactly",
                "you are right", "you're right", "thats right", "that's right",
                "sure", "of course",
            )
            no_words = (
                "no", "nope", "nah", "not really", "incorrect",
                "wrong", "don't think so", "dont think so",
            )
            end_words = (
                "stop", "quit", "exit", "end the game", "end game",
                "goodbye", "bye", "that's enough", "thats enough",
            )
            if any(w in lower for w in yes_words):
                local_state["is_yes"] = True
            if any(w in lower for w in no_words):
                local_state["is_no"] = True
            if any(w in lower for w in end_words):
                local_state["wants_end"] = True
            if "ready" in lower:
                local_state["is_ready"] = True
        
        true_flags = [k for k, v in local_state.items() if v]
        # If we got a clear signal (not both yes and no), return without an LLM call.
        if true_flags and not (local_state["is_yes"] and local_state["is_no"]):
            return local_state
        
        # Slow path by LLM (fallback)
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
        
        if cls.model == "gemini":
            prompt = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in messages
            )
            
            response = cls.llm_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "temperature": 0.0,
                    "response_mime_type": "application/json",
                }
            )
            content = response.text.strip()
        else:
            completion = cls.llm_client.chat.completions.create(
                model=cls.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=50,
            )
            content = (completion.choices[0].message.content or "").strip()
        
        data = Game._coerce_json(content)
        if not isinstance(data, dict):
            if cls.logger:
                cls.logger.log_line("[warn] Classifier non-JSON; Raw=" + repr((content or "")[:500]))
            data = {}
        
        state = dict(default_state)
        if isinstance(data, dict):
            for k in state.keys():
                if k in data:
                    state[k] = bool(data[k])
        return state

    @classmethod
    def _parse_llm_output(cls, data) -> tuple[str, list, list]:
        """
        Parse the JSON from the main game LLM into:
        - robot utterance
        - profile (character facts)
        - most_likely (cleaned candidates)
        """
        if not isinstance(data, dict):
            text = str(data)
            cls.working_memory["profile"] = []
            cls.working_memory["most_likely"] = []
            return text, []

        response_text = data.get("response") or ""
        profile = data.get("profile") or []
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
        return response_text, profile, cleaned_candidates

    @classmethod
    def _furhat_say(cls, text: str):
        print("Furhat:", text)
        cls.furhat_client.request_speak_text(text)
    
    @classmethod
    def _furhat_backchannel(cls):
        """Quick acknowledgement after the user's answer to reduce perceived latency."""
        try:
            phrase = random.choice(BACKCHANNELS)
        except IndexError:
            return
        cls._furhat_say(phrase)

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
            cls.game_end_reason = GameEndReason.USER_QUIT
            return nr + 1, "end"

        if turn_type == "start":
            if is_ready:
                return nr + 1, "normal"
            else:
                return nr + 1, "start"

        if turn_type == "guess":
            if is_yes:
                cls.game_end_reason = GameEndReason.CORRECT_GUESS
                return nr + 1, "end"
            if question_count >= MAX_TURNS:
                cls.game_end_reason = GameEndReason.MAX_QUESTIONS_REACHED
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
    def _update_working_memory(cls, turn_key: str, furhat_question: str, user_answer: str, profile: list, most_likely: list):
        cls.working_memory["turns"][turn_key] = {
            "furhat_question": furhat_question,
            "user_answer": user_answer,
            "profile_snapshot": profile,
            "most_likely_snapshot": most_likely,
        }

    @classmethod
    def run(cls, model=DEFAULT_LLM, host="127.0.0.1", auth_key=None, context_filler=False):
        ok = cls.init(model=model, host=host, auth_key=auth_key, context_filler=context_filler)
        if not ok:
            return

        current_turn = 1
        current_type = "start"

        try:
            # Game loop
            while True:
                current_turn = 1
                current_type = "start"
                
                # Single round
                while current_type != "end":
                    current_turn, current_type = cls.turn(
                        nr=current_turn,
                        turn_type=current_type,
                    )

                # Say contextual goodbye
                end_message = get_end_message(cls.game_end_reason)
                cls._furhat_say(end_message)
                
                if cls.game_end_reason == GameEndReason.USER_QUIT:
                    break
                
                # Ask about replay
                replay_message = get_replay_message()
                cls._furhat_say(replay_message)
                
                # Listen and classify reply
                user_utt = cls._furhat_listen()
                user_state = cls._interpret_user(replay_message, user_utt)
                
                # Clear ambiguous classification once: ask again if ambiguous
                if not (user_state.get("is_yes") or user_state.get("is_no") or user_state.get("wants_end")):
                    clarification = "I didn't catch that. Please say 'yes' to play again or 'no' to quit."
                    cls._furhat_say(clarification)
                    user_utt = cls._furhat_listen()
                    user_state = cls._interpret_user(clarification, user_utt)
                    
                # Decide whether to replay
                if user_state.get("is_yes"):
                    cls.working_memory = create_working_memory()
                    cls.working_memory["start_intro_done"] = True
                    cls.context_filler_cached = None
                    cls.game_end_reason = None
                    cls._furhat_say("Great! Think of a new character and say 'ready' when you are ready.")
                    continue
                else:
                    cls._furhat_say("Okay, thanks for playing. Goodbye!")
                    break


            
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
    parser.add_argument("--context_filler", action="store_true")
    args = parser.parse_args()

    Game.run(
        model=args.model,
        host=args.host,
        auth_key=args.auth_key,
        context_filler=args.context_filler,
    )