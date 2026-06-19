from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


def load_paper_records(data_dir: str | Path = "data") -> list[tuple[Path, dict[str, Any]]]:
    records: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(Path(data_dir).glob("*.json")):
        if path.name == "id_mapping.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or not data.get("title"):
            continue
        embedding = data.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            continue
        records.append((path, data))
    return records


def build_faiss(
    data_dir: str | Path = "data",
    index_path: str | Path | None = None,
    mapping_path: str | Path | None = None,
) -> dict[str, Any]:
    if faiss is None:
        raise RuntimeError("FAISS 未安装，请先安装 faiss-cpu。")

    data_dir = Path(data_dir)
    index_path = Path(index_path or data_dir / "faiss.index")
    mapping_path = Path(mapping_path or data_dir / "id_mapping.json")
    records = load_paper_records(data_dir)
    if not records:
        raise ValueError("没有找到包含 embedding 的论文数据。")

    dimension = len(records[0][1]["embedding"])
    valid_records = [
        (path, data)
        for path, data in records
        if len(data["embedding"]) == dimension
    ]
    vectors = np.asarray(
        [data["embedding"] for _, data in valid_records],
        dtype=np.float32,
    )
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    mapping = {
        str(index_id): {
            "title": data["title"],
            "json_file": path.name,
            "authors": data.get("authors", []),
        }
        for index_id, (path, data) in enumerate(valid_records)
    }

    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_index = index_path.with_suffix(".index.tmp")
    temporary_mapping = mapping_path.with_suffix(".json.tmp")
    faiss.write_index(index, str(temporary_index))
    temporary_mapping.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2),
        encoding="utf8",
    )
    temporary_index.replace(index_path)
    temporary_mapping.replace(mapping_path)

    return {
        "paper_count": index.ntotal,
        "dimension": dimension,
        "index_path": str(index_path),
        "mapping_path": str(mapping_path),
        "skipped_count": len(records) - len(valid_records),
    }


if __name__ == "__main__":
    print(build_faiss())
