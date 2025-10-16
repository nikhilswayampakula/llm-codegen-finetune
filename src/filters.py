from __future__ import annotations
from pathlib import Path
EXT_ALLOW = {'.py', '.js', '.ts', '.java', '.go', '.rs', '.cpp', '.c', '.cs'}
EXCLUDE_DIRS = {'node_modules','dist','build','.git','.venv','__pycache__','target'}

def want_file(p: Path) -> bool:
    if p.suffix.lower() not in EXT_ALLOW: return False
    for parent in p.parents:
        if parent.name.lower() in EXCLUDE_DIRS: return False
    try:
        if p.stat().st_size > 200_000: return False  # skip huge files
    except Exception:
        return False
    return True
