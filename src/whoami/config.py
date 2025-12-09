class Config():
    DEFAULT_LLM = "gpt-4o-mini" 
    GUESS_THRESHOLD = 0.8   # Furhat will guess when it thinks his guess has 80%+ chance of succeeding.
    MAX_TURNS = 15
    MIN_QUESTIONS_BEFORE_GUESS = 7
    BACKCHANNEL_PROB = 0.5  # probability of backchannel utterance on user response