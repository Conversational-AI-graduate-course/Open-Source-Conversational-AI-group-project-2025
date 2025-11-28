from openai import OpenAI
import logging
import os
from dotenv import load_dotenv
from furhat_realtime_api import FurhatClient
import argparse
import time
import json
import signal


GAME_SYSTEM_PROMPT = """
“Play the “Who Am I?” game. In the game, your interlocutor will assign you a character.
For example, a famous person or a character from a cartoon. Your task is to guess it.
For this, you will ask yes-or-no questions. Choose questions that give as much information as possible.
For example, “Is the person female”, “Is the person alive?”. You cannot ask more than 20 questions.
Ask new questions based on the information you already have. If you need more information,
you can ask your interlocutor for a hint instead of a yes-or-no question.
Try to make guesses as soon as you have gathered enough information. If you guess right, you win.
If the guess is wrong, you can continue asking questions until you reach the limit of 20.
Once the game has finished, meaning that you have asked 20 questions or guessed which character has been assigned to you,
you will ask whether your interlocutor wants to try it another time or stop here.
"""


def create_game_state():
    return {
        "questions_asked": [],
        "answers": [],
        "hints": [],
        "question_count": 0,
        "game_active": True,
    }


class InteractionLogger:
    def __init__(self):
        # assumes root = Open-Source-Conversational-AI-group-project-2025
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        logs_dir = os.path.join(base_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        self.path = os.path.join(logs_dir, f"log-{ts}.txt")
        self.f = open(self.path, "a", encoding="utf-8", buffering=1)
        self.t0 = time.monotonic()

        self._line_sep = "-" * 47    # aesthetics

    def _write(self, s: str):
        try:
            self.f.write(s.rstrip("\n") + "\n")
            self.f.flush()
            os.fsync(self.f.fileno())
        except Exception:
            pass

    def _stamp(self) -> str:
        elapsed = int(time.monotonic() - self.t0)
        return f"[{elapsed // 60:02d}:{elapsed % 60:02d}]"

    def say(self, who: str, text: str):
        self._write(f'{self._stamp()} {who}: "{str(text)}"')

    def stats(self, turn_stats: str, full_prompt: str):
        self._write(self._line_sep)
        self._write(turn_stats.strip())
        self._write("Following LLM prompt:")
        self._write(full_prompt)
        self._write(self._line_sep)

    def close(self):
        try:
            self._write(self._stamp() + " [logger] closing")
            self.f.close()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--host", type=str, default="127.0.0.1", help="Furhat robot IP address")
        parser.add_argument("--auth_key", type=str, default=None, help="Authentication key for Realtime API")
        args = parser.parse_args()

        load_dotenv(override=True)

        openai = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        furhat = FurhatClient(host=args.host, auth_key=args.auth_key)
        furhat.set_logging_level(logging.INFO)


        # Our local logger
        logger = InteractionLogger()
        def _handle_signal(sig, frame):
            logger._write(logger._stamp() + f" [signal] {sig} received")
            logger.close()
            raise SystemExit(0)

        for _sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(_sig, _handle_signal)
            except Exception:
                pass


        try:
            furhat.connect()
        except Exception as e:
            print(f"Failed to connect to Furhat on {args.host}.")
            raise SystemExit(0)
        
        furhat.request_voice_config('Danielle-Neural (en-US) - Amazon Polly')


        model = "gpt-3.5-turbo"  

        # First utterance/greeting
        robot_utt = "Hello, I am Furhat, the The Who Am I player. Let's play, please assign me a character"

        furhat.request_attend_user() # Default to attending the closest user

        game_state = create_game_state()


        while True:
            game_state = create_game_state()
            messages = [{"role": "system", "content": GAME_SYSTEM_PROMPT}]

            # Game loop
            while game_state["game_active"]:

                print("Robot:", robot_utt)
                logger.say("Furhat", robot_utt)  # LOG
                messages.append({"role": "assistant", "content": robot_utt})
                furhat.request_speak_text(robot_utt)

                # Listen to user
                user_utt = furhat.request_listen_start()
                print("User:", user_utt)
                logger.say("User", user_utt)  # LOG
                messages.append({"role": "user", "content": user_utt})

                # Classify answer or hint
                if user_utt.strip().lower() in ["yes", "no"]:
                    game_state["answers"].append(user_utt)
                else:
                    game_state["hints"].append(user_utt)

                # Memory blob for model reasoning
                memory_blob = f"""
    Game state:
    - Questions asked: {game_state['questions_asked']}
    - Answers: {game_state['answers']}
    - Hints: {game_state['hints']}
    - Question count: {game_state['question_count']}
    """
                print("Memory blob:" + memory_blob)
                messages.append({"role": "assistant", "content": memory_blob})


                try:
                    yes_count = sum(1 for a in game_state['answers'] if str(a).strip().lower() == "yes")
                    no_count = sum(1 for a in game_state['answers'] if str(a).strip().lower() == "no")
                    turn_stats_summary = (
                        f"Stats Turn {game_state['question_count'] + 1}: totals: Q={game_state['question_count']}, yes={yes_count}, no={no_count}, hints={len(game_state['hints'])}"
                    )
                except Exception:
                    turn_stats_summary = "Stats Turn ?: totals: (unavailable)"
                
                full_prompt_text = json.dumps(messages, ensure_ascii=False, indent=2)
                logger.stats(turn_stats_summary, full_prompt_text)  # LOG
                
                response = openai.chat.completions.create(
                    model=model,
                    messages=messages,
                    )
                robot_utt = response.choices[0].message.content.strip()


                # Check if model guessed early
                if robot_utt.lower().startswith(("i guess", "are you thinking of", "my guess is")):
                    furhat.request_speak_text(robot_utt)
                    print("Robot guess:", robot_utt)
                    logger.say("Furhat", robot_utt)  # LOG

                    furhat.request_speak_text("Was my guess correct?")
                    correctness = furhat.request_listen_start().lower().strip()
                    print("User:", correctness)
                    logger.say("User", correctness)  # LOG

                    if "yes" in correctness:
                        furhat.request_speak_text("Great! I guessed it correctly!")
                        game_state["game_active"] = False
                        break
                    else:
                        furhat.request_speak_text("Okay, I'll keep trying!")
                        # Don't count this as a question
                        game_state["questions_asked"].append(robot_utt)
                        continue

                game_state["questions_asked"].append(robot_utt)
                game_state["question_count"] += 1

                # Question limit
                if game_state["question_count"] > 20:
                    game_state["game_active"] = False

                    info = "I have reached 20 questions! Time for my final guess."
                    print("Robot:", info)
                    logger.say("Furhat", info)  # LOG
                    furhat.request_speak_text(info)
                    messages.append({"role": "assistant", "content": info})

                    # Memory prompt for the final guess, had to add bullet points otherwise it would give general answers
                    final_prompt = f"""
    You must now make your FINAL GUESS.
    - One sentence only
    - Format: "I guess the character is <NAME> because <reason>." 

    Memory:
    Questions: {game_state['questions_asked']}
    Answers: {game_state['answers']}
    Hints: {game_state['hints']}
    """ 
                    messages.append({"role": "assistant", "content": final_prompt})

                    try:
                        turn_stats_summary_final = (
                            f"Stats FINAL: totals: Q={game_state['question_count']}, "
                            f"yes={yes_count}, no={no_count}, hints={len(game_state['hints'])}"
                        )
                    except Exception:
                        turn_stats_summary_final = "Stats FINAL: totals: (unavailable)"
                    logger.stats(turn_stats_summary_final, json.dumps(messages, ensure_ascii=False, indent=2))  # LOG

                    final_resp = openai.chat.completions.create(
                         model=model,
                         messages=messages,
                         temperature=0.2
                     )

                    final_guess = final_resp.choices[0].message.content.strip()
                    
                    print("Robot (final guess):", final_guess)
                    logger.say("Furhat", final_guess)  # LOG
                    furhat.request_speak_text(final_guess)

                    user_utt = furhat.request_listen_start()
                    print("User:", user_utt)
                    logger.say("User", user_utt)  # LOG
                    messages.append({"role": "user", "content": user_utt})

                    logger.stats("post-final user reply", json.dumps(messages, ensure_ascii=False, indent=2))  # LOG
                    response = openai.chat.completions.create(
                        model=model,
                        messages=messages,
                    )
                    robot_utt = response.choices[0].message.content.strip()
                    logger.say("Furhat", robot_utt)  # LOG
                    furhat.request_speak_text(robot_utt)
                    break

            # Outside game loop: play again?
            time.sleep(1.0)
            furhat.request_speak_text("Would you like to play again? Say yes to continue or no to stop.")
            logger.say("Furhat", "Would you like to play again? Say yes to continue or no to stop.")  # LOG
            user_utt = furhat.request_listen_start().lower()
            logger.say("User", user_utt)  # LOG

            if "yes" in user_utt:
                furhat.request_speak_text("Great! Let's play again.")
                robot_utt = "Think of a character"
                continue  # restart outer loop
            else:
                furhat.request_speak_text("Okay! Thanks for playing. Goodbye.")
                break

        pass
    except KeyboardInterrupt:
        # So logs still finish.
        pass
    finally:
        try:
            logger.close()
        except Exception:
            pass