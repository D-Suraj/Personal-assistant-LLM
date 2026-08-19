import sqlite3
from pathlib import Path
from typing import Any


class WorkMindDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    content TEXT NOT NULL,
                    last_modified TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS work_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    activity TEXT NOT NULL,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_files_project_path
                    ON files(project_id, path);
                """
            )

    @staticmethod
    def _rows(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
        return [dict(row) for row in rows]

    def list_projects(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM projects ORDER BY created_at DESC"
            ).fetchall()
        return self._rows(rows)

    def create_project(self, name: str, description: str = "") -> int:
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO projects (name, description) VALUES (?, ?)",
                (name.strip(), description.strip()),
            )
            return int(cur.lastrowid)

    def update_project(self, project_id: int, name: str, description: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE projects SET name = ?, description = ? WHERE id = ?",
                (name.strip(), description.strip(), project_id),
            )

    def delete_project(self, project_id: int) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM files WHERE project_id = ?", (project_id,))
            conn.execute("DELETE FROM work_logs WHERE project_id = ?", (project_id,))
            conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))

    def delete_file(self, project_id: int, file_path: str) -> None:
        self.delete_files(project_id, [file_path])

    def delete_files(self, project_id: int, file_paths: list[str]) -> None:
        if not file_paths:
            return
        with self.connect() as conn:
            placeholders = ",".join("?" for _ in file_paths)
            conn.execute(
                f"DELETE FROM files WHERE project_id = ? AND path IN ({placeholders})",
                (project_id, *file_paths),
            )

    def list_files(self, project_id: int) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, project_id, name, path, last_modified
                FROM files
                WHERE project_id = ?
                ORDER BY path
                """,
                (project_id,),
            ).fetchall()
        return self._rows(rows)

    def upsert_file(
        self,
        project_id: int,
        name: str,
        path: str,
        content: str,
        last_modified: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO files (project_id, name, path, content, last_modified)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(project_id, path) DO UPDATE SET
                    name = excluded.name,
                    content = excluded.content,
                    last_modified = excluded.last_modified
                """,
                (project_id, name, path, content, last_modified),
            )

    def read_file(self, project_id: int, path_or_name: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT name, path, content, last_modified
                FROM files
                WHERE project_id = ? AND (path = ? OR name = ?)
                LIMIT 1
                """,
                (project_id, path_or_name, path_or_name),
            ).fetchone()
        return dict(row) if row else None

    def recent_files(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT p.name as project, f.name as file, f.path, f.last_modified
                FROM files f
                JOIN projects p ON f.project_id = p.id
                ORDER BY f.last_modified DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return self._rows(rows)
