from __future__ import annotations
import argparse, tempfile, shutil
from pathlib import Path
from git import Repo
from src.filters import want_file

def clone_and_collect(urls_file: str, out_dir: str):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    urls = [u.strip() for u in Path(urls_file).read_text().splitlines() if u.strip() and not u.startswith('#')]
    for url in urls:
        with tempfile.TemporaryDirectory() as tmp:
            print(f"Cloning {url} ...")
            Repo.clone_from(url, tmp, depth=1)
            t = Path(tmp)
            files = [p for p in t.rglob('*') if p.is_file() and want_file(p)]
            for p in files:
                rel = p.relative_to(t)
                tgt = out / rel
                tgt.parent.mkdir(parents=True, exist_ok=True)
                try:
                    txt = p.read_text(encoding='utf-8', errors='ignore')
                except Exception:
                    continue
                tgt.write_text(txt, encoding='utf-8')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--urls', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    clone_and_collect(args.urls, args.out)

if __name__ == '__main__':
    main()
