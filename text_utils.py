from datetime import datetime, timezone
from pathlib import Path


TEXT_EXTS = {
    ".bat",
    ".c",
    ".cpp",
    ".cs",
    ".css",
    ".dockerfile",
    ".env",
    ".go",
    ".graphql",
    ".h",
    ".hpp",
    ".html",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".pdf",
    ".mdx",
    ".php",
    ".prisma",
    ".py",
    ".rb",
    ".rs",
    ".sass",
    ".scss",
    ".sh",
    ".sql",
    ".svelte",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".vue",
    ".xml",
    ".yaml",
    ".yml",
}

SKIP_DIRS = {
    ".git",
    ".hg",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "coverage",
    "data",
    "dist",
    "node_modules",
    "venv",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_text_path(path: Path) -> bool:
    name = path.name.lower()
    if name in {"dockerfile", "makefile", ".gitignore"}:
        return True
    return path.suffix.lower() in TEXT_EXTS


def read_text_file(path: Path, max_chars: int = 200_000) -> str:
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="ignore")
    return text[:max_chars]


def iter_text_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and is_text_path(path):
            files.append(path)
    return files
