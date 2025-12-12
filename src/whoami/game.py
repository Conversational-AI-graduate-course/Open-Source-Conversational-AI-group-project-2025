from openai import OpenAI
from google import genai
from google.genai import types
from enum import Enum
from furhat_realtime_api import FurhatClient
from .prompts import PROMPTS
from .logger import InteractionLogger
from .json_parser import JsonParser
from .backchannels import BACKCHANNELS
from dotenv import load_dotenv
import os
import re
import time
import json
import random
import threading

# Main Game
class Game:
    """
    Playing Who Am I? using Furhat robot fueled by LLM.
    """
    
    @staticmethod
    def create_working_memory():
        """Working memory for the game."""
        return {
            "turns": {},                # "Turn 1": {"furhat_question": "", "user_answer": "","profile_snapshot": [...]  "most_likely_snapshot": [...]}
            "question_count": 0,        # how many question-type turns we've done
            "profile": [],              # summary of gathered information
            "most_likely": [],          # current top candidates
            "start_intro_done": False,  # LLM intro was already delivered once
            "context_fillers": [],      # memory of N last fillers
        }

    llm_client = None
    furhat_client = None
    model = None
    working_memory = create_working_memory()
    logger = None
    context_filler_mode = False
    context_filler_cached = None
    

    @classmethod
    def init(cls, config, model, host="127.0.0.1", auth_key=None, context_filler=False):
        load_dotenv(override=True)
        cls.config = config
        cls.model = config.DEFAULT_LLM if not model else model
        cls.working_memory = cls.create_working_memory()
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
            cls.furhat_client.request_face_config(
                face_id="adult - Isabel",
                visibility=True,
                microexpressions=True
            )
            print("[log] Connected to Furhat (via Realtime API)")
        except Exception:
            print("[error] Failed to connect to Furhat.")
            print("[tip] Is Virtual Furhat running on this PC and Remote API enabled?")
            print("[tip] If using Virtual Furhat locally, keep --host 127.0.0.1.")
            print("[tip] If an auth key is required, pass --auth_key or set FURHAT_AUTH_KEY in .env.")
            return False

        return True
    
    
    class GameEndReason(Enum):
        USER_QUIT = "user_quit"
        CORRECT_GUESS = "correct_guess"
        MAX_QUESTIONS_REACHED = "max_questions_reached"
        
    
    # Change messages here
    @classmethod
    def get_end_message(cls, end_reason):
        """Return contextual end message based on how the game ended."""
        if end_reason == cls.GameEndReason.CORRECT_GUESS:
            return "Wuhuu, I guessed correctly! Thank you for playing with me!"
        elif end_reason == cls.GameEndReason.MAX_QUESTIONS_REACHED:
            return "Oh no! I've used all my questions and couldn't guess correctly. That was a tough one!"
        else:  # USER_QUIT
            return "Thank you for playing the Who am I game with me. Goodbye!"
        
    @staticmethod
    def get_replay_message():
        return "Would you like to play another round?"

    @classmethod
    def _get_normal_prompt(cls) -> str:
        """Get the normal question prompt, possibly with context filler."""
        base = PROMPTS["NORMAL"]
        if cls.context_filler_mode:
            # Ask the LLM to also propose a contextual filler"
            return (
                base
                + PROMPTS["CONTEXT_FILLER"]
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
        memory_blob_used = None
        if turn_type == "start" and start_intro_done_before:
            robot_utterance = ""
            profile = cls.working_memory.get("profile", [])
            most_likely = cls.working_memory.get("most_likely", [])
        else:

        # If we have prefetched filler, say it while generating question
            if turn_type == "normal" and cls.context_filler_mode and cls.context_filler_cached:
                context_filler = cls.context_filler_cached  
                cls.context_filler_cached = None

                # Store context filler in memory blob
                cls._remember_context_filler(context_filler)
                
                # Start saying the filler in parallel
                filler_thread = threading.Thread(
                    target=cls._furhat_say, 
                    args=(context_filler,), 
                    daemon=True
                )
                filler_thread.start()
                
                # While filler is being said generate the question 
                memory_blob_used = cls._format_memory_blob()
                messages = [
                    {"role": "system", "content": PROMPTS["SYSTEM"]},
                    {"role": "user", "content": cls._format_memory_blob()},
                    {"role": "user", "content": prompt},
                ]
                llm_json = cls._call_llm(messages)
                
                
                robot_utterance, profile, most_likely = JsonParser.parse_llm_output(llm_json)
                cls.working_memory["profile"] = profile
                cls.working_memory["most_likely"] = most_likely
                
                # Wait for filler to finish
                filler_thread.join()
                time.sleep(0.2) #200 ms pause before question

            else:
                memory_blob_used = cls._format_memory_blob()
                messages = [
                    {"role": "system", "content": PROMPTS["SYSTEM"]},
                    {"role": "user", "content": cls._format_memory_blob()},
                    {"role": "user", "content": prompt},
                ]

                # 2) call LLM (JSON response)
                llm_json = cls._call_llm(messages)

                # 3) parse LLM JSON output -> robot utterance + updated profile and most_likely
                robot_utterance, profile, most_likely = JsonParser.parse_llm_output(llm_json)
                cls.working_memory["profile"] = profile
                cls.working_memory["most_likely"] = most_likely

            # 4) Furhat speaks
            cls._furhat_say(robot_utterance)
            if turn_type == "start":
                cls.working_memory["start_intro_done"] = True

        # 5) listen to user
        t_listen_start = time.monotonic()
        user_utterance = cls._furhat_listen()
        listen_elapsed = time.monotonic() - t_listen_start

        if turn_type == "normal" and (user_utterance or "").strip():
            if not cls.context_filler_mode and random.random() < cls.config.BACKCHANNEL_PROB: # added to not use it if using the context_filler
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
                    "memory_blob_sent_to_llm": memory_blob_used,
                    "context_fillers_memory": list(cls.working_memory.get("context_fillers", [])),
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
    def _remember_context_filler(cls, filler: str, max_items: int = 5, max_chars: int = 120):
        """Store last N fillers, trimmed. Style-only memory."""
        if not isinstance(filler, str):
            return
        s = " ".join(filler.split()).strip()
        if not s:
            return
        s = s[:max_chars]  # trim to short length

        lst = cls.working_memory.setdefault("context_fillers", [])
        lst.append(s)
        # keep only last N
        if len(lst) > max_items:
            del lst[:-max_items]

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

        fillers = cls.working_memory.get("context_fillers", [])
        if fillers:
            lines.append("")
            lines.append("Previous context fillers (STYLE ONLY; not facts; avoid repeating these phrases):")
            for i, f in enumerate(fillers[-5:], 1):
                lines.append(f"- {i}) {f}")


        profile = cls.working_memory.get("profile", [])
        if profile:
            lines.append("Character profile (confirmed facts):")
            for item in profile:
                if isinstance(item, str):
                    lines.append(item)
                elif isinstance(item, dict):
                    lines.append(f"{item.get('name','?')}: {item.get('why','')}")
                else:
                    lines.append(str(item))
            lines.append("")

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
            cls.game_end_reason = cls.GameEndReason.USER_QUIT
            return nr + 1, "end"

        if turn_type == "start":
            if is_ready:
                return nr + 1, "normal"
            else:
                return nr + 1, "start"

        if turn_type == "guess":
            if is_yes:
                cls.game_end_reason = cls.GameEndReason.CORRECT_GUESS
                return nr + 1, "end"
            if question_count >= cls.config.MAX_TURNS:
                cls.game_end_reason = cls.GameEndReason.MAX_QUESTIONS_REACHED
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
            top_likelihood >= cls.config.GUESS_THRESHOLD
            and turn_type not in ("guess", "end")
            and question_count >= MIN_QUESTIONS_BEFORE_GUESS
        ):
            return nr + 1, "guess"

        # Force final guess
        if question_count >= cls.config.MAX_TURNS and turn_type not in ("guess", "end"):
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
    def run(cls, config, model=None, host="127.0.0.1", auth_key=None, context_filler=False):
        if model is None:
            model = cls.config.DEFAULT_LLM
            
        ok = cls.init(config=config, model=model, host=host, auth_key=auth_key, context_filler=context_filler)
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
                end_message = cls.get_end_message(cls.game_end_reason)
                cls._furhat_say(end_message)
                
                if cls.game_end_reason == cls.GameEndReason.USER_QUIT:
                    break
                
                # Ask about replay
                replay_message = cls.get_replay_message()
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
                    cls.working_memory = cls.create_working_memory()
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
