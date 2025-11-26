from openai import OpenAI
import logging
import os
from dotenv import load_dotenv
from furhat_realtime_api import FurhatClient
import argparse
import time
import json


EXPRESSIONS = {
    "happy": "Smile",
    "sad": "ExpressSad",
    "anger": "ExpressAnger",
    "disgust": "ExpressDisgust",
    "browraise": "BrowRaise",
    "fear": "ExpressFear",
    "close eyes": "CloseEyes",
    "open eyes": "OpenEyes",
}


def demo_show_faces(furhat: FurhatClient, pause_after_each: float = 5.0, intensity: float = 1.0, duration: float = 1.0):
    """
    For showcasing (some) facial expressions with logs. Run python main.py --demo_show_faces
    """

    expressions = list(dict.fromkeys(EXPRESSIONS.values()))
    
    print(f"[DEMO] Expressions should start in {pause_after_each} sec.")
    for name in expressions:
        time.sleep(pause_after_each)
        print(f"[DEMO] Gesture: {name}")
        try:
            furhat.request_gesture_start(
                name = name,
                intensity = intensity,
                duration = duration,
                wait = True
            )
        except Exception as e:
            print(f"[DEMO] Gesture '{name}' failed: {e}")


def demo_instruct_faces(furhat: FurhatClient, pause_after_each: float = 0.2, intensity: float = 1.0, duration: float = 1.0):
    """
    For instructing the robot to show (some) facial expressions. Run python main.py --demo_instruct_faces
    """

    first_four = list(EXPRESSIONS.items())[:4] # Pick from just the first four expressions
    human_terms = [k for k, _ in first_four]
    allowed_codes = [v for _, v in first_four]

    question = f"Which facial expression should I do? Choose from {', '.join(human_terms)}."
    system_prompt = (
        "You are controlling a social robot playing a short expression game.\n"
        "Rules:\n"
        "1) ALWAYS respond with a single JSON object and NOTHING ELSE.\n"
        '2) The JSON must have keys "response" (robot line) and "face".\n'
        f'3) "face" must be exactly one of: {json.dumps(allowed_codes)}.\n'
        "4) Keep 'response' short and friendly (<= 20 words).\n"
    )

    messages = [{"role": "developer", "content": system_prompt}]


    furhat.request_attend_user()



    robot_utt = question
    model = "gpt-3.5-turbo"

    while True:
        print("Robot:", robot_utt)
        try:
            furhat.request_speak_text(robot_utt, True, True)
        except Exception as e:
            print(f"[WARN] speak prompt failed: {e}")

        messages.append({"role": "assistant", "content": robot_utt})

        user_utt = furhat.request_listen_start()
        print("User:", user_utt if isinstance(user_utt, str) else str(user_utt))
        messages.append({"role": "user", "content": user_utt})

        try:
            completion = openai.chat.completions.create(
                model = model,
                messages = messages,
                response_format = {"type": "json_object"},
                temperature = 0.2,
            )
            content = completion.choices[0].message.content
            data = json.loads(content)
        except Exception as e:
            print(f"[ERROR] LLM/json error: {e}")
            data = {"response": "Please choose happy, sad, anger, or disgust.", "face": "Smile"}

        response_text = data.get("response") or "Got it."
        face_value = data.get("face") or "Smile"

        if face_value not in allowed_codes:
            mapped = EXPRESSIONS.get(face_value.lower())
            face_value = mapped if mapped in allowed_codes else "Smile"

        try:
            furhat.request_gesture_start(name=face_value, intensity=intensity, duration=duration, wait=True)
        except Exception as e:
            print(f"[WARN] gesture failed: {e}")

        try:
            furhat.request_speak_text(response_text, True, True)
        except Exception as e:
            print(f"[ERROR] speak response failed: {e}")

        robot_utt = "Want to try another one? " + question


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Furhat robot IP address")
    parser.add_argument("--auth_key", type=str, default=None, help="Authentication key for Realtime API")
    parser.add_argument("--demo_show_faces", action="store_true", help="Run a show facial-expression demo and exit")
    parser.add_argument("--demo_instruct_faces", action="store_true", help="Run a instruct facial-expression demo")
    args = parser.parse_args()

    load_dotenv(override=True)

    openai = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    furhat = FurhatClient(host=args.host, auth_key=args.auth_key)
    furhat.set_logging_level(logging.INFO)


    try:
        furhat.connect()
    except Exception as e:
        print(f"Failed to connect to Furhat on {args.host}.")
        raise SystemExit(0)
    
    furhat.request_voice_config('Danielle-Neural (en-US) - Amazon Polly')


    # Demo instruct facial expressions ----

    if args.demo_instruct_faces:
        try:
            demo_instruct_faces(
                furhat, 
                pause_after_each = 0.2, 
                intensity = 1.0, 
                duration = 1.0
            )
        finally:
            try:
                furhat.disconnect()
            finally:
                print("Disconnected.")
        raise SystemExit(0)
    # Run demo show faces if no argument passed
    else:
        try:
            demo_show_faces(
                furhat, 
                pause_after_each = 5.0, 
                intensity = 1.0, 
                duration = 1.0
            )
        finally:
            try:
                furhat.disconnect()
            finally:
                print("Disconnected.")
        raise SystemExit(0)
    
