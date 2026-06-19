from __future__ import annotations

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from paper_library import PaperLibrary


def test_list_existing_library() -> None:
    library = PaperLibrary()
    papers = library.list_papers()
    assert len(papers) >= 20
    assert any("Mamba" in paper.title for paper in papers)
    assert library.stats()["index_exists"]


def test_rebuild_faiss_in_temporary_library() -> None:
    source_files = [
        path
        for path in sorted(Path("data").glob("*.json"))
        if path.name != "id_mapping.json"
    ][:3]

    with TemporaryDirectory() as directory:
        data_dir = Path(directory) / "data"
        pdf_dir = Path(directory) / "papers"
        data_dir.mkdir()
        for source in source_files:
            shutil.copy2(source, data_dir / source.name)

        library = PaperLibrary(data_dir=data_dir, pdf_dir=pdf_dir)
        result = library.rebuild_index()

        assert result["paper_count"] == len(source_files)
        assert result["dimension"] == 512
        assert (data_dir / "faiss.index").exists()
        mapping = json.loads(
            (data_dir / "id_mapping.json").read_text(encoding="utf8")
        )
        assert len(mapping) == len(source_files)


if __name__ == "__main__":
    test_list_existing_library()
    test_rebuild_faiss_in_temporary_library()
    print("paper library tests passed")
