from openai import OpenAI
import logging
import os
from dotenv import load_dotenv
from furhat_realtime_api import FurhatClient
import argparse
import time


def demo_faces(furhat: FurhatClient, pause_after_each: float = 5.0, intensity: float = 1.0, duration: float = 1.0):
    """
    For showcasing (some) facial expressions with logs. Run python main.py --demo_faces
    """

    expressions = [
        "Smile",
        "BrowRaise",
        "ExpressAnger",
        "ExpressSad",
        "ExpressFear",
        "ExpressDisgust",
        "CloseEyes",
        "OpenEyes",
    ]
    
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Furhat robot IP address")
    parser.add_argument("--auth_key", type=str, default=None, help="Authentication key for Realtime API")
    parser.add_argument("--demo_faces", action="store_true", help="Run a facial-expression demo and exit")
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
    
    # Demo facial expressions ----

    if args.demo_faces:
        try:
            demo_faces(
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
    
    # ----------------------------

    model = "gpt-3.5-turbo"  
    system_prompt = "You are a friendly robot looking for a nice little chat."
    messages = [{"role": "developer", "content": system_prompt}] 
    robot_utt = "Hello, I am Furhat. How are you today?"

    furhat.request_attend_user() # This will default to attending the closest user

    while True:
        print("Robot: ", robot_utt)
        messages.append({"role": "assistant", "content": robot_utt})
        furhat.request_speak_text(robot_utt)
        user_utt = furhat.request_listen_start()
        print("User: ", user_utt)
        messages.append({"role": "user", "content": user_utt})
        response = openai.chat.completions.create(model=model, messages=messages)
        robot_utt = response.choices[0].message.content
