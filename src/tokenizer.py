import re
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_tok = TweetTokenizer(reduce_len=True, strip_handles=True)

EXTRA = {
    # contractions sklearn misses
    "u", "ur", "im", "ive", "youre", "dont", "doesnt", "cant",
    "wont", "isnt", "wasnt", "didnt", "wouldnt", "couldnt",
    "thats", "its", "hes", "shes", "theyre", "weve",

    # pure filler
    "ok", "okay", "yeah", "yep", "nope", "nah",
    "oh", "ah", "uh", "um", "eh", "pls", "plz",
    "gonna", "gotta", "wanna", "tbh", "rly",

    # cross-class neutral bleed
    "just", "like", "time", "come", "win", "need",
    "xd", "guys", "lol", "omg", "thx", "rng",
    "play", "game", "tank", "spot",
    "today", "people", "kids", "nice", "sry",
    "yes", "gl", "try", "let", "stay", "got", "left", "line",
    "elc", "ebr", "wg", "wot", "tier", "doing", "loose",

    # Dota-specific neutral
    "ggwp", "mmr", "ff", "dc", "w8", "rc", "hf",
    "reconnecting", "reconnect", "lag", "end",
    "haha", "rofl", "ty", "thanks", "said", "min", "sec",
    "wr", "team", "afk", "wait", "pro", "wow", "good",

    # Dota hero names — zero toxicity signal
    "abaddon", "alchemist", "axe", "bane", "batrider", "beastmaster",
    "bloodseeker", "brewmaster", "bristleback", "broodmother", "centaur",
    "chen", "clinkz", "clockwerk", "dazzle", "dawnbreaker", "disruptor",
    "doom", "enchantress", "enigma", "grimstroke", "gyrocopter", "hoodwink",
    "huskar", "invoker", "io", "jakiro", "juggernaut", "kunkka", "leshrac",
    "lich", "lifestealer", "lina", "lion", "luna", "lycan", "magnus",
    "marci", "mars", "medusa", "meepo", "mirana", "morphling", "muerta",
    "necrophos", "oracle", "pangolier", "phoenix", "puck", "pudge", "pugna",
    "razor", "riki", "rubick", "silencer", "slark", "slardar", "snapfire",
    "sniper", "spectre", "sven", "techies", "tidehunter", "timbersaw",
    "tinker", "tiny", "tusk", "underlord", "undying", "ursa", "viper",
    "visage", "warlock", "weaver", "zeus",

    # hero abbreviations
    "sf", "qop", "ta", "am", "es", "pa", "pl", "wk", "dk",
    "lc", "tb", "sk", "ns", "od", "wd", "sd", "ss", "kotl",
    "mk", "vs", "dp", "ck", "bh", "sb", "ember", "void", "storm",
    "jugg", "jug", "drow", "bara", "gyro", "legion", "rosh", "clock",
    "necro", "invo", "potm",
}

STOPWORDS = ENGLISH_STOP_WORDS.union(EXTRA)


def tokenize(text):
    text = re.sub(r'(\w)\*+(\w)', r'\1\2', str(text))  # f**k → fk, before tokenizing
    # tokenizization using TweetTokenizer, which is designed for social media text and handles emojis, punctuation, etc.
    tokens = _tok.tokenize(text)
    # preserve ALLCAPS for expressive intensity (NOOB, WTF), lowercase rest
    # WORD
    tokens = [w + "_CAPS" if w.isupper() and len(w) > 1 else w.lower() for w in tokens]
    tokens = [w for w in tokens if w not in STOPWORDS]
    tokens = [w for w in tokens if re.search(r"\w", w)]
    return tokens
