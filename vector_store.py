import hashlib
from pathlib import Path
from typing import Any

import chromadb


class VectorStore:
    def __init__(self, persist_dir: Path):
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir / "chroma"))
        self.collection = self.client.get_or_create_collection(
            "workmind_chunks", metadata={"hnsw:space": "cosine"}
        )

    @staticmethod
    def _id(project_id: int, file_path: str, chunk_index: int, text: str) -> str:
        digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]
        return f"{project_id}:{file_path}:{chunk_index}:{digest}"

    def upsert_chunks(
        self,
        project_id: int,
        file_path: str,
        file_name: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        ids = [
            self._id(project_id, file_path, index, text)
            for index, text in enumerate(chunks)
        ]
        where = {
            "$and": [
                {"project_id": {"$eq": str(project_id)}},
                {"file_path": {"$eq": file_path}},
            ]
        }
        existing = self.collection.get(where=where, include=[])
        existing_ids = set(existing.get("ids") or [])
        metadatas = [
            {
                "project_id": str(project_id),
                "file_path": file_path,
                "file_name": file_name,
                "chunk_index": index,
            }
            for index in range(len(chunks))
        ]
        if chunks:
            self.collection.upsert(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        stale_ids = existing_ids.difference(ids)
        if stale_ids:
            self.collection.delete(ids=sorted(stale_ids))
        return len(chunks)

    def search(
        self,
        query_embedding: list[float],
        project_id: int | None = None,
        top_k: int = 6,
    ) -> list[dict[str, Any]]:
        where = {"project_id": str(project_id)} if project_id else None
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )
        rows: list[dict[str, Any]] = []
        for index, document in enumerate(results.get("documents", [[]])[0]):
            rows.append(
                {
                    "document": document,
                    "metadata": results.get("metadatas", [[]])[0][index],
                    "distance": results.get("distances", [[]])[0][index],
                    "id": results.get("ids", [[]])[0][index],
                }
            )
        return rows

    def delete_project(self, project_id: int) -> None:
        self.collection.delete(where={"project_id": str(project_id)})

    def delete_file(self, project_id: int, file_path: str) -> None:
        self.delete_files(project_id, [file_path])

    def delete_files(self, project_id: int, file_paths: list[str]) -> None:
        if not file_paths:
            return
        if len(file_paths) == 1:
            where: dict[str, Any] = {
                "$and": [
                    {"project_id": {"$eq": str(project_id)}},
                    {"file_path": {"$eq": file_paths[0]}},
                ]
            }
        else:
            where = {
                "$and": [
                    {"project_id": {"$eq": str(project_id)}},
                    {"file_path": {"$in": file_paths}},
                ]
            }
        self.collection.delete(where=where)
