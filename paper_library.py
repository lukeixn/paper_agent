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


@dataclass
class DeleteResult:
    title: str
    success: bool
    json_file: str = ""
    deleted_files: list[str] | None = None
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

    def delete_paper(
        self,
        paper: Paper | str | Path,
        *,
        rebuild_index: bool = True,
        delete_pdf: bool = True,
    ) -> tuple[DeleteResult, dict[str, Any] | None]:
        paper_obj = self._resolve_paper(paper)
        json_path = self._paper_json_path(paper_obj)
        deleted_files: list[str] = []

        if not json_path.exists():
            return (
                DeleteResult(
                    title=paper_obj.title,
                    success=False,
                    json_file=str(json_path),
                    deleted_files=[],
                    message="论文 JSON 文件不存在。",
                ),
                None,
            )

        matching_pdf_paths = self._matching_pdf_paths(paper_obj, json_path)
        json_path.unlink()
        deleted_files.append(str(json_path))

        if delete_pdf:
            for pdf_path in matching_pdf_paths:
                pdf_path.unlink(missing_ok=True)
                deleted_files.append(str(pdf_path))

        index_result = self._refresh_index_after_delete() if rebuild_index else None
        return (
            DeleteResult(
                title=paper_obj.title,
                success=True,
                json_file=str(json_path),
                deleted_files=deleted_files,
                message="删除成功。",
            ),
            index_result,
        )

    def delete_many(
        self,
        papers: list[Paper | str | Path],
        *,
        delete_pdf: bool = True,
    ) -> tuple[list[DeleteResult], dict[str, Any] | None]:
        results: list[DeleteResult] = []
        deleted = False
        for paper in papers:
            result, _ = self.delete_paper(
                paper,
                rebuild_index=False,
                delete_pdf=delete_pdf,
            )
            results.append(result)
            deleted = deleted or result.success

        index_result = self._refresh_index_after_delete() if deleted else None
        return results, index_result

    def _resolve_paper(self, paper: Paper | str | Path) -> Paper:
        if isinstance(paper, Paper):
            return paper

        value = str(paper)
        for candidate in self.list_papers():
            if value in {
                candidate.title,
                candidate.source_file,
                Path(candidate.source_file).name,
            }:
                return candidate

        path = self._safe_data_path(value)
        try:
            data = json.loads(path.read_text(encoding="utf8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"无法定位论文：{value}") from exc
        return Paper.from_dict(data, source_file=str(path))

    def _paper_json_path(self, paper: Paper) -> Path:
        if paper.source_file:
            return self._safe_data_path(paper.source_file)
        filename = PaperParser.sanitize_filename(paper.title) + ".json"
        return self.data_dir / filename

    def _safe_data_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        data_root = self.data_dir.resolve()

        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            direct_resolved = candidate.resolve()
            if data_root == direct_resolved or data_root in direct_resolved.parents:
                resolved = direct_resolved
            else:
                resolved = (self.data_dir / candidate).resolve()

        if data_root != resolved and data_root not in resolved.parents:
            raise ValueError(f"论文路径不在数据目录内：{path}")
        return resolved

    def _matching_pdf_paths(
        self,
        paper: Paper,
        json_path: Path,
    ) -> list[Path]:
        names = {
            PaperParser.sanitize_filename(paper.title) + ".pdf",
            json_path.with_suffix(".pdf").name,
        }
        try:
            data = json.loads(json_path.read_text(encoding="utf8"))
            pdf_filename = data.get("pdf_filename")
            if isinstance(pdf_filename, str) and pdf_filename.strip():
                names.add(Path(pdf_filename).name)
        except (OSError, json.JSONDecodeError):
            pass

        matches: list[Path] = []
        for name in names:
            path = self.pdf_dir / name
            if path.exists() and path.is_file():
                matches.append(path)
        return sorted(set(matches))

    def _refresh_index_after_delete(self) -> dict[str, Any] | None:
        if self.list_papers():
            return self.rebuild_index()

        removed: list[str] = []
        for path in [
            self.data_dir / "faiss.index",
            self.data_dir / "id_mapping.json",
        ]:
            if path.exists():
                path.unlink()
                removed.append(str(path))
        return {
            "paper_count": 0,
            "dimension": 0,
            "index_path": str(self.data_dir / "faiss.index"),
            "mapping_path": str(self.data_dir / "id_mapping.json"),
            "skipped_count": 0,
            "removed_files": removed,
        }

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
            self._record_pdf_filename(paper.title, safe_pdf_name)
            shutil.copy2(temporary_path, final_pdf_path)
            return paper
        finally:
            temporary_path.unlink(missing_ok=True)

    def _record_pdf_filename(self, title: str, pdf_filename: str) -> None:
        json_path = self.data_dir / f"{PaperParser.sanitize_filename(title)}.json"
        try:
            data = json.loads(json_path.read_text(encoding="utf8"))
        except (OSError, json.JSONDecodeError):
            return
        data["pdf_filename"] = Path(pdf_filename).name
        temporary_path = json_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf8",
        )
        temporary_path.replace(json_path)

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
