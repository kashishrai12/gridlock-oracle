
import re
import pandas as pd

# concept -> trigger terms (English + Kannada + transliterations seen in the data)
KEYWORD_GROUPS = {
    "txt_blockage":  ["block", "blocked", "both lane", "road clos", "closure", "jam",
                      "ರಸ್ತೆ ಬಂದ್", "ಬ್ಲಾಕ್", "ಸಂಚಾರ ಬಂದ್", "ಬಂದ್"],
    "txt_treefall":  ["tree fall", "tree fell", "fallen tree", "tree", "ಮರ", "ಮರ ಬಿದ್ದಿದೆ"],
    "txt_accident":  ["accident", "collision", "collide", "overturn", "capsize", "hit",
                      "ಅಪಘಾತ", "ಡಿಕ್ಕಿ", "ಪಲ್ಟಿ"],
    "txt_breakdown": ["breakdown", "break down", "off road", "not start", "starting problem",
                      "clutch", "tyre", "tire", "puncture", "engine", "gear",
                      "ವಾಹನ ಆಫ್", "ಬ್ರೇಕ್ ಡೌನ್", "ಆಫ್ ಆಗಿ"],
    "txt_waterlog":  ["water", "flood", "rain", "logging", "logged",
                      "ನೀರು", "ಮಳೆ", "ಮಳೆ ನೀರು"],
    "txt_diversion": ["divert", "diversion", "reroute", "re-route", "ಡೈವರ್ಶನ್", "ಬೇರೆ ದಾರಿ"],
    "txt_heavy":     ["heavy", "container", "trailer", "lorry", "truck", "tanker",
                      "ಭಾರಿ", "ಲಾರಿ", "ಟ್ರಕ್"],
    "txt_fire":      ["fire", "smoke", "burning", "ಬೆಂಕಿ", "ಹೊಗೆ"],
}

TEXT_FEATURES = list(KEYWORD_GROUPS.keys()) + ["txt_word_count", "txt_has_text"]


def add_text_features(df, col="description"):
    """Return a DataFrame (aligned to df.index) of interpretable text features."""
    if col in df.columns:
        s = df[col].astype(str).fillna("")
    else:
        s = pd.Series([""] * len(df), index=df.index)
    s = s.replace({"nan": "", "None": "", "none": ""})
    norm = s.str.casefold()

    out = pd.DataFrame(index=df.index)
    for fname, terms in KEYWORD_GROUPS.items():
        pat = "|".join(re.escape(t.casefold()) for t in terms)
        out[fname] = norm.str.contains(pat, regex=True, na=False).astype(int)
    out["txt_word_count"] = s.str.split().apply(len).astype(float)
    out["txt_has_text"] = (s.str.strip() != "").astype(int)
    return out


def extract_text_features(text):
    """Single-string version for inference (returns a dict)."""
    df = pd.DataFrame({"description": [text if text is not None else ""]})
    row = add_text_features(df).iloc[0]
    return {k: (int(v) if k != "txt_word_count" else float(v)) for k, v in row.items()}