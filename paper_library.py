from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, BinaryIO

from academic_search import AcademicSearchService, SearchCandidate
from paper_parser import PaperInfo, PaperParser
from schemas import Paper
from vector_store.build_faiss import build_faiss


@dataclass
class ImportResult:
    filename: str
    success: bool
    title: str = ""
    message: str = ""


class PaperLibrary:
    def __init__(
        self,
        data_dir: str | Path = "data",
        pdf_dir: str | Path = "papers",
    ):
        self.data_dir = Path(data_dir)
        self.pdf_dir = Path(pdf_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

    def list_papers(self) -> list[Paper]:
        papers: list[Paper] = []
        for path in sorted(self.data_dir.glob("*.json")):
            if path.name == "id_mapping.json":
                continue
            try:
                data = json.loads(path.read_text(encoding="utf8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict) or not data.get("title"):
                continue
            papers.append(Paper.from_dict(data, source_file=str(path)))
        return sorted(papers, key=lambda paper: paper.title.lower())

    def stats(self) -> dict[str, Any]:
        papers = self.list_papers()
        dimensions = {len(paper.embedding) for paper in papers if paper.embedding}
        return {
            "paper_count": len(papers),
            "embedding_dimension": next(iter(dimensions)) if len(dimensions) == 1 else 0,
            "index_exists": (self.data_dir / "faiss.index").exists(),
            "mapping_exists": (self.data_dir / "id_mapping.json").exists(),
        }

    def rebuild_index(self) -> dict[str, Any]:
        return build_faiss(data_dir=self.data_dir)

    def import_pdf(
        self,
        uploaded_file: BinaryIO,
        filename: str,
        *,
        model_config: dict[str, Any],
        overwrite: bool = False,
        parser: PaperParser | None = None,
    ) -> PaperInfo:
        safe_pdf_name = PaperParser.sanitize_filename(Path(filename).stem) + ".pdf"
        final_pdf_path = self.pdf_dir / safe_pdf_name
        if final_pdf_path.exists() and not overwrite:
            raise FileExistsError(f"PDF 已存在：{safe_pdf_name}")

        suffix = Path(filename).suffix or ".pdf"
        with NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            shutil.copyfileobj(uploaded_file, temporary)
            temporary_path = Path(temporary.name)

        try:
            parser = parser or PaperParser(
                save_dir=self.data_dir,
                model_config=model_config,
            )
            paper = parser.parse_pdf(
                temporary_path,
                save_json=True,
                overwrite=overwrite,
            )
            shutil.copy2(temporary_path, final_pdf_path)
            return paper
        finally:
            temporary_path.unlink(missing_ok=True)

    def import_many(
        self,
        uploaded_files: list[Any],
        *,
        model_config: dict[str, Any],
        overwrite: bool = False,
    ) -> tuple[list[ImportResult], dict[str, Any] | None]:
        results: list[ImportResult] = []
        imported = False
        parser = PaperParser(
            save_dir=self.data_dir,
            model_config=model_config,
        )
        for uploaded_file in uploaded_files:
            try:
                uploaded_file.seek(0)
                paper = self.import_pdf(
                    uploaded_file,
                    uploaded_file.name,
                    model_config=model_config,
                    overwrite=overwrite,
                    parser=parser,
                )
                results.append(
                    ImportResult(
                        filename=uploaded_file.name,
                        success=True,
                        title=paper.title,
                        message="解析并保存成功",
                    )
                )
                imported = True
            except Exception as exc:
                results.append(
                    ImportResult(
                        filename=uploaded_file.name,
                        success=False,
                        message=str(exc),
                    )
                )

        index_result = self.rebuild_index() if imported else None
        return results, index_result

    def import_search_candidates(
        self,
        candidates: list[SearchCandidate],
        *,
        model_config: dict[str, Any],
        overwrite: bool = False,
    ) -> tuple[list[ImportResult], dict[str, Any] | None]:
        if len(candidates) > 10:
            raise ValueError("单次最多导入 10 篇论文。")

        search_service = AcademicSearchService()
        parser = PaperParser(
            save_dir=self.data_dir,
            model_config=model_config,
        )
        results: list[ImportResult] = []
        imported = False

        for candidate in candidates:
            try:
                pdf_buffer = search_service.download_pdf(candidate)
                paper = self.import_pdf(
                    pdf_buffer,
                    pdf_buffer.name,
                    model_config=model_config,
                    overwrite=overwrite,
                    parser=parser,
                )
                paper_path = (
                    self.data_dir
                    / f"{PaperParser.sanitize_filename(paper.title)}.json"
                )
                paper_data = json.loads(paper_path.read_text(encoding="utf8"))
                paper_data.update(
                    {
                        "doi": candidate.doi,
                        "source_url": candidate.landing_page_url,
                        "discovery_source": "OpenAlex",
                    }
                )
                paper_path.write_text(
                    json.dumps(paper_data, ensure_ascii=False, indent=2),
                    encoding="utf8",
                )
                results.append(
                    ImportResult(
                        filename=pdf_buffer.name,
                        success=True,
                        title=paper.title,
                        message="下载、解析并保存成功",
                    )
                )
                imported = True
            except Exception as exc:
                results.append(
                    ImportResult(
                        filename=candidate.title,
                        success=False,
                        message=str(exc),
                    )
                )

        index_result = self.rebuild_index() if imported else None
        return results, index_result
