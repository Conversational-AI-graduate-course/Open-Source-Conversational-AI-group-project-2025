
# Furhat "Who am I?" Game - Realtime API (Python)

This repository contains a Python-based environment for interacting with the **Furhat Realtime API**.

Quick overview:
- Game settings:
    - DEFAULT_LLM, GUESS_THRESHOLD, MAX_TURNS
    - PROMPTS dict with SYSTEM, START, NORMAL, HINT, GUESS, END, CLASSIFY prompts
- Working memory:
    - create_working_memory(): per-turn Q/A and running most_likely candidates
- Logging:
    - InteractionLogger: timestamped logs of each turn (LLM input/output + user replies)
- Main Game class:
    - Game.init(): loads .env, connects OpenAI + Furhat, resets memory and logger
    - Game.turn(): one full turn (prompt LLM → Furhat speaks → user listens → LLM classifies user → decide next turn → log)
    - Game._decide_next_turn(): early-guess logic, MAX_TURNS cap, and "ready" handling
    - Game.run(): main loop starting in "start" mode and running turns until "end", then says goodbye
- Run section (CLI):
    - Parses optional --host, --auth_key, --model and calls Game.run...

## Project Structure
```
project-root/
│
├── src/
│   └── whoami/
|       └── backchannels.py      # List of filler/backchannel responses during the game
|       └── config.py            # Set LLM and Game flow settings
|       └── game.py              # Core game logic 
|       └── json_parser.py       # Utility for validating model-generated JSON
|       └── logger.py            # Utility for debugging and tracing game events
│       └── main.py              # Entrypoint script
|       └── prompts.py           # Prompt templates and system messages for the LLM
│   └── demo.py              # Facial expression demos
│   └── main-old-version.py  # Old single file script
│
├── requirements.txt
├── Dockerfile
└── README.md
``` 


## Running using Python Virtual Environment on Windows

You can run the project directly on Windows using a Python virtual environment.

---

### 1. Install Python

Download and install Python 3.11 or later:
[https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/)

Make sure to check **“Add Python to PATH”** during installation.

---

### 2. Create a Virtual Environment

Open **PowerShell** or **Command Prompt** inside the project folder:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\activate
```

You should now see something like:

```
(venv) C:\path\to\project>
```

---

### 3. Install Requirements

With the venv active:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 4. Run the Furhat Script

```powershell
python .\src\main.py
```

---

### 5. Deactivate the Environment

```powershell
deactivate
```

---

### Notes for Windows Users

* If `pip install` fails due to build tools, install **Microsoft C++ Build Tools** (rarely needed).
---

## (Optional) Run using Docker

### 1. Install Docker
#### **Windows / macOS**

1. Download **Docker Desktop**:
   [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
2. Install it normally.
3. Make sure Docker Desktop is running (you should see the whale icon).

#### **Linux (Ubuntu / Debian)**

```bash
sudo apt update
sudo apt install docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
# log out and back in
```

---

### 2. Build the Docker Image

Open a terminal **inside the project folder** (the one containing the Dockerfile), then run:

```bash
docker build -t furhat-app .
```

This will:

* Pull the Python 3.11 slim image
* Install requirements
* Copy your code
* Prepare everything to run your main Python script

---

### 3. Run the Project in Docker

To start the application:

```bash
docker run --rm -it -v .:/app/ --network=host furhat-app
```

If your application needs access to local files (e.g., logs or data), you can mount a folder:

```bash
docker run --rm -it -v "$(pwd)/data:/app/data" --network-host furhat-app
```

---

### 4. Verifying Furhat Realtime API Works

```bash
docker run --rm -it furhat-app python -c "import furhat; print('Furhat OK')"
```

---

### 5. Stopping Containers

If you ever start a container without `--rm`, list running containers:

```bash
docker ps
```

Then stop one:

```bash
docker stop <container_id>
```

---

## Contributing

Feel free to open issues, suggest improvements, or submit PRs.

---

## Additional Resources

* Furhat Realtime API: [https://pypi.org/project/furhat-realtime-api/](https://pypi.org/project/furhat-realtime-api/)
* Example Python code from Furhat:
  [https://github.com/FurhatRobotics/realtime-api-examples/tree/main/python](https://github.com/FurhatRobotics/realtime-api-examples/tree/main/python)
* Docker documentation: [https://docs.docker.com/](https://docs.docker.com/)
* OpenAI API (Chat Completion): [https://platform.openai.com/docs/api-reference/chat/create](https://platform.openai.com/docs/api-reference/chat/create)
* Gemini API: [http://googleapis.github.io/python-genai/](http://googleapis.github.io/python-genai/)
