"""
check_text_fields.py — assess whether free-text columns are rich enough to justify NLP.
Run: python check_text_fields.py --data data/flipkart_gridlock.csv
"""
import argparse
import pandas as pd

TEXT_COLS = ["description", "comment", "reason_breakdown", "cargo_material"]


def assess(df, col):
    if col not in df.columns:
        return f"  {col:18s}: (column not present)"
    s = df[col].astype(str).str.strip()
    s = s.replace({"nan": "", "None": "", "none": ""})
    nonblank = s[s != ""]
    n = len(df)
    fill = len(nonblank) / n if n else 0
    uniq = nonblank.nunique()
    uniq_ratio = uniq / len(nonblank) if len(nonblank) else 0
    avg_words = nonblank.str.split().apply(len).mean() if len(nonblank) else 0
    verdict = ("RICH — good NLP candidate" if fill > 0.4 and uniq_ratio > 0.3 and avg_words >= 3
               else "WEAK — skip (too sparse/repetitive)" if fill < 0.2 or uniq < 20
               else "MAYBE — borderline")
    return (f"  {col:18s}: fill={fill:5.1%} | unique={uniq:5d} ({uniq_ratio:4.0%}) "
            f"| avg_words={avg_words:4.1f} | {verdict}")


def main(path):
    df = pd.read_csv(path)
    print(f"\n[rows: {len(df)}]  free-text column assessment:\n")
    for c in TEXT_COLS:
        print(assess(df, c))
    print("\n--- sample values (non-blank) ---")
    for c in TEXT_COLS:
        if c in df.columns:
            s = df[c].astype(str).str.strip()
            s = s[~s.str.lower().isin(["nan", "none", ""])]
            if len(s):
                print(f"\n{c} (showing up to 5):")
                for v in s.drop_duplicates().head(5):
                    print(f"   • {v[:120]}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    main(ap.parse_args().data)