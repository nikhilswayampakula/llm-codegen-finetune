from __future__ import annotations
import argparse, json, re
from pathlib import Path

PY_PATTERN = r'^(def |class ).*:'
JS_PATTERN = r'^(function |class |const |let |var ).*'

def split_into_examples(code: str, suffix: str):
    pattern = PY_PATTERN if suffix == '.py' else JS_PATTERN
    lines = code.splitlines()
    out, cur = [], []
    for ln in lines:
        if re.match(pattern, ln):
            if cur: out.append('\n'.join(cur)); cur = []
        cur.append(ln)
    if cur: out.append('\n'.join(cur))
    pairs = []
    for ch in out:
        ls = ch.splitlines()
        if not ls: continue
        prompt = ls[0] + '\n'
        completion = '\n'.join(ls[1:]) + '\n'
        if len(completion.strip()) >= 10:
            pairs.append({'prompt': prompt, 'completion': completion})
    return pairs

def build_dataset(src_dir: str, out_jsonl: str):
    out = Path(out_jsonl); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for p in Path(src_dir).rglob('*'):
            if not p.is_file(): continue
            try:
                txt = p.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                continue
            pairs = split_into_examples(txt, p.suffix.lower())
            for ex in pairs:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    print(f"Wrote dataset to {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    build_dataset(args.src, args.out)

if __name__ == '__main__':
    main()
