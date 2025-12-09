import os
import argparse
from .game import Game
from .config import Config


if __name__ == "__main__":
    config = Config()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--auth_key", type=str, default=os.getenv("FURHAT_AUTH_KEY"))
    parser.add_argument("--model", type=str, default=config.DEFAULT_LLM)
    parser.add_argument("--context_filler", action="store_true")
    args = parser.parse_args()

    Game.run(
        config=config,
        model=args.model,
        host=args.host,
        auth_key=args.auth_key,
        context_filler=args.context_filler,
    )