from __future__ import annotations

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from paper_library import PaperLibrary
from paper_parser import PaperParser


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


def test_delete_paper_removes_record_pdf_and_rebuilds_index() -> None:
    source_files = [
        path
        for path in sorted(Path("data").glob("*.json"))
        if path.name != "id_mapping.json"
    ][:3]

    with TemporaryDirectory() as directory:
        data_dir = Path(directory) / "data"
        pdf_dir = Path(directory) / "papers"
        data_dir.mkdir()
        pdf_dir.mkdir()
        for source in source_files:
            shutil.copy2(source, data_dir / source.name)

        library = PaperLibrary(data_dir=data_dir, pdf_dir=pdf_dir)
        library.rebuild_index()
        paper = library.list_papers()[0]
        pdf_path = pdf_dir / (
            PaperParser.sanitize_filename(paper.title) + ".pdf"
        )
        pdf_path.write_bytes(b"%PDF-1.4\n")

        result, index_result = library.delete_paper(paper)

        assert result.success
        assert not Path(result.json_file).exists()
        assert not pdf_path.exists()
        assert index_result is not None
        assert index_result["paper_count"] == len(source_files) - 1
        assert len(library.list_papers()) == len(source_files) - 1


def test_delete_last_paper_clears_stale_index_files() -> None:
    source = next(
        path
        for path in sorted(Path("data").glob("*.json"))
        if path.name != "id_mapping.json"
    )

    with TemporaryDirectory() as directory:
        data_dir = Path(directory) / "data"
        pdf_dir = Path(directory) / "papers"
        data_dir.mkdir()
        pdf_dir.mkdir()
        shutil.copy2(source, data_dir / source.name)

        library = PaperLibrary(data_dir=data_dir, pdf_dir=pdf_dir)
        library.rebuild_index()
        assert (data_dir / "faiss.index").exists()
        assert (data_dir / "id_mapping.json").exists()

        results, index_result = library.delete_many(library.list_papers())

        assert [result.success for result in results] == [True]
        assert index_result is not None
        assert index_result["paper_count"] == 0
        assert not (data_dir / "faiss.index").exists()
        assert not (data_dir / "id_mapping.json").exists()


if __name__ == "__main__":
    test_list_existing_library()
    test_rebuild_faiss_in_temporary_library()
    test_delete_paper_removes_record_pdf_and_rebuilds_index()
    test_delete_last_paper_clears_stale_index_files()
    print("paper library tests passed")
