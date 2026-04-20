from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

EXTRA = {
    # informal contractions sklearn misses
    "u", "ur", "im", "ive", "youre", "dont", "doesnt", "cant",
    "wont", "isnt", "wasnt", "didnt", "wouldnt", "couldnt",
    "thats", "its", "hes", "shes", "theyre", "weve",

    # pure filler
    "ok", "okay", "yeah", "yep", "nope", "nah",
    "oh", "ah", "uh", "um", "eh", "pls", "plz",
    "gonna", "gotta", "wanna", "tbh", "rly",
    "xd", "lol", "omg", "thx", "sry", "hf",

    # cross-class neutral bleed
    "just", "like", "really", "guys", "know", "come",
    "want", "time", "got", "win", "play", "game",
    "end", "nice", "good", "wait", "pro", "wow",
    "haha", "rofl", "ty", "thanks", "said", "min",
    "sec", "wr", "need", "team", "afk",

    # Dota hero full names
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
    "visage", "warlock", "weaver", "zeus", "kez", "largo",

    # hero chat abbreviations
    "sf", "qop", "ta", "am", "es", "pa", "pl", "wk", "dk",
    "lc", "tb", "sk", "ns", "od", "wd", "sd", "ss", "kotl",
    "mk", "vs", "dp", "ck", "bh", "sb", "ember", "void", "storm",
    "jugg", "jug", "drow", "bara", "gyro", "legion", "rosh", "clock",
    "necro", "invo", "potm",

    # game mechanics with zero signal
    "ggwp", "mmr", "ff", "dc", "w8", "rc",
    "pause", "unpause", "commend", "commended",
    "reconnecting", "reconnect", "lag",
}

STOPWORDS = list(ENGLISH_STOP_WORDS.union(EXTRA))
