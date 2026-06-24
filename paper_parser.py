from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, TypeVar

import fitz
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from models.langchain_llm import OfflineLLM, get_llm


StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


class PaperInfo(BaseModel):
    title: str
    authors: list[str]
    abstract: str
    summary: str
    keywords: list[str]
    contributions: list[str]
    limitations: list[str]
    embedding: list[float]
    doi: str = ""
    source_url: str = ""
    discovery_source: str = ""


class PaperInfoTmp(BaseModel):
    title: str
    authors: list[str]
    abstract: str
    summary: str
    keywords: list[str]
    contributions: list[str]
    limitations: list[str]


class PaperChunkNotes(BaseModel):
    section_summary: str
    title_candidates: list[str] = Field(default_factory=list)
    author_candidates: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    results: list[str] = Field(default_factory=list)
    contributions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class PaperParser:
    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-small-zh-v1.5",
        save_dir: str | Path = "data",
        model_config: dict[str, Any] | None = None,
        chunk_size: int = 12000,
    ):
        self.model_config = model_config or {}
        self.llm = get_llm(
            provider=self.model_config.get("provider"),
            api_key=self.model_config.get("api_key"),
            model_name=self.model_config.get("model_name"),
            base_url=self.model_config.get("base_url"),
            temperature=self.model_config.get("temperature"),
        )
        if isinstance(self.llm, OfflineLLM):
            raise RuntimeError("解析 PDF 需要配置可用的在线 LLM。")

        self.embedding_model = SentenceTransformer(
            embedding_model_name,
            local_files_only=True,
        )
        self.output_parser = PydanticOutputParser(
            pydantic_object=PaperInfoTmp
        )
        self.chunk_parser = PydanticOutputParser(
            pydantic_object=PaperChunkNotes
        )
        self.chunk_size = chunk_size
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        filename = re.sub(r'[<>:"/\\|?*]', "_", filename).strip().rstrip(".")
        return filename or "untitled-paper"

    @staticmethod
    def read_pdf_pages(pdf_path: str | Path) -> list[str]:
        path = Path(pdf_path)
        with fitz.open(path) as document:
            pages = [page.get_text().strip() for page in document]

        pages = [page for page in pages if page]
        if not pages:
            raise ValueError(
                "PDF 中没有提取到文本，可能是扫描件或受保护文档。"
            )
        return pages

    @classmethod
    def read_pdf(cls, pdf_path: str | Path) -> str:
        return "\n\n".join(cls.read_pdf_pages(pdf_path))

    @staticmethod
    def chunk_pages(
        pages: list[str],
        max_chars: int = 12000,
    ) -> list[str]:
        """Group every page into ordered chunks without dropping content."""
        chunks: list[str] = []
        current_parts: list[str] = []
        current_length = 0

        def flush() -> None:
            nonlocal current_parts, current_length
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_length = 0

        for page_number, page in enumerate(pages, start=1):
            page_text = f"[Page {page_number}]\n{page}"
            if len(page_text) > max_chars:
                flush()
                for start in range(0, len(page_text), max_chars):
                    chunks.append(page_text[start : start + max_chars])
                continue

            separator_length = 2 if current_parts else 0
            if current_parts and (
                current_length + separator_length + len(page_text) > max_chars
            ):
                flush()
            current_parts.append(page_text)
            current_length += separator_length + len(page_text)

        flush()
        return chunks

    @staticmethod
    def build_embedding_text(paper: PaperInfoTmp) -> str:
        return "\n".join(
            [
                f"标题：{paper.title}",
                f"摘要：{paper.abstract}",
                f"总结：{paper.summary}",
                f"关键词：{' '.join(paper.keywords)}",
                f"主要贡献：{' '.join(paper.contributions)}",
            ]
        )

    @staticmethod
    def _strip_markdown_fence(content: str) -> str:
        text = content.strip()
        fence_match = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return fence_match.group(1).strip() if fence_match else text

    @staticmethod
    def _extract_json_object(content: str) -> str:
        text = PaperParser._strip_markdown_fence(content)
        if not text:
            raise ValueError("模型返回为空，无法解析 JSON。")

        start = text.find("{")
        if start < 0:
            raise ValueError("模型返回中没有 JSON 对象。")

        in_string = False
        escaped = False
        depth = 0
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]

        raise ValueError("模型返回中的 JSON 对象不完整。")

    @staticmethod
    def _parse_structured_content(
        content: str,
        parser: PydanticOutputParser,
        model_type: type[StructuredModel],
    ) -> StructuredModel:
        try:
            json_text = PaperParser._extract_json_object(content)
            json_text = re.sub(r",\s*([}\]])", r"\1", json_text)
            payload = json.loads(json_text)
        except Exception as json_error:
            raise ValueError(
                "模型返回不是完整合法 JSON，且无法从文本中提取可用 JSON。"
            ) from json_error
        try:
            return model_type.model_validate(payload)
        except Exception as validation_error:
            raise ValueError(
                "模型返回 JSON 字段不符合论文解析 schema。"
            ) from validation_error

    def invoke_structured(
        self,
        prompt: str,
        parser: PydanticOutputParser,
        model_type: type[StructuredModel],
    ) -> StructuredModel:
        response = self.llm.invoke(prompt)
        try:
            return self._parse_structured_content(
                response.content,
                parser,
                model_type,
            )
        except Exception as parse_error:
            repair_prompt = f"""
下面的模型输出没有通过 JSON/schema 校验。请只把它修复成合法 JSON 对象，不要输出 Markdown，不要解释。

目标格式：
{parser.get_format_instructions()}

原始输出：
{response.content}

错误信息：
{parse_error}
"""
            repaired = self.llm.invoke(repair_prompt)
            return self._parse_structured_content(
                repaired.content,
                parser,
                model_type,
            )

    def extract_chunk_notes(
        self,
        chunk: str,
        chunk_index: int,
        chunk_count: int,
    ) -> PaperChunkNotes:
        prompt = f"""
你正在逐段阅读一篇完整论文。这是第 {chunk_index}/{chunk_count} 个文本块。
请只根据当前文本块提取后续汇总所需的研究证据。

{self.chunk_parser.get_format_instructions()}

要求：
1. section_summary 使用中文，准确概括当前块。
2. 保留具体方法、实验结果、贡献和作者明确承认的局限。
3. 如果当前块没有某类信息，返回空列表，不要猜测。
4. 标题和作者只有在文本中明确出现时才记录。

当前文本块：
{chunk}
"""
        return self.invoke_structured(
            prompt,
            self.chunk_parser,
            PaperChunkNotes,
        )

    def synthesize_paper_info(
        self,
        notes: list[PaperChunkNotes],
    ) -> PaperInfoTmp:
        notes = self.compact_notes(notes)
        notes_payload = json.dumps(
            [note.model_dump() for note in notes],
            ensure_ascii=False,
        )
        prompt = f"""
你已经阅读了一篇论文的全部文本块。请根据下面所有分块笔记，生成整篇论文
的最终结构化信息。

{self.output_parser.get_format_instructions()}

要求：
1. summary、contributions 和 limitations 使用中文。
2. keywords 返回 5 到 10 个关键词。
3. 综合所有文本块，覆盖研究问题、核心方法、实验设置、主要结果和结论。
4. 只使用分块笔记中有依据的信息，不要虚构。
5. 合并重复内容，但不要遗漏只在论文后半部分出现的实验结果或局限。

全文分块笔记：
{notes_payload}
"""
        return self.invoke_structured(
            prompt,
            self.output_parser,
            PaperInfoTmp,
        )

    def compact_notes(
        self,
        notes: list[PaperChunkNotes],
        max_payload_chars: int = 32000,
    ) -> list[PaperChunkNotes]:
        """Hierarchically compress notes without excluding any text block."""
        current = notes
        while len(
            json.dumps(
                [note.model_dump() for note in current],
                ensure_ascii=False,
            )
        ) > max_payload_chars:
            batches: list[list[PaperChunkNotes]] = []
            batch: list[PaperChunkNotes] = []
            batch_size = 0
            for note in current:
                note_size = len(
                    json.dumps(note.model_dump(), ensure_ascii=False)
                )
                if batch and batch_size + note_size > max_payload_chars:
                    batches.append(batch)
                    batch = []
                    batch_size = 0
                batch.append(note)
                batch_size += note_size
            if batch:
                batches.append(batch)

            if len(batches) >= len(current):
                break
            current = [
                self.merge_note_batch(batch, index, len(batches))
                for index, batch in enumerate(batches, start=1)
            ]
        return current

    def merge_note_batch(
        self,
        notes: list[PaperChunkNotes],
        batch_index: int,
        batch_count: int,
    ) -> PaperChunkNotes:
        prompt = f"""
请压缩论文分块笔记的第 {batch_index}/{batch_count} 组，为最终全文汇总保留
所有重要证据。

{self.chunk_parser.get_format_instructions()}

要求：
1. 合并重复内容。
2. 不遗漏方法、实验结果、贡献和局限。
3. 不添加原笔记中没有的信息。

待压缩笔记：
{json.dumps([note.model_dump() for note in notes], ensure_ascii=False)}
"""
        return self.invoke_structured(
            prompt,
            self.chunk_parser,
            PaperChunkNotes,
        )

    def extract_paper_info(self, text: str) -> PaperInfoTmp:
        """Compatibility entry point that still processes all supplied text."""
        chunks = [
            text[start : start + self.chunk_size]
            for start in range(0, len(text), self.chunk_size)
        ]
        notes = [
            self.extract_chunk_notes(chunk, index, len(chunks))
            for index, chunk in enumerate(chunks, start=1)
        ]
        return self.synthesize_paper_info(notes)

    def extract_full_paper_info(self, pages: list[str]) -> PaperInfoTmp:
        chunks = self.chunk_pages(pages, self.chunk_size)
        notes = [
            self.extract_chunk_notes(chunk, index, len(chunks))
            for index, chunk in enumerate(chunks, start=1)
        ]
        return self.synthesize_paper_info(notes)

    def generate_embedding(self, paper: PaperInfoTmp) -> list[float]:
        embedding = self.embedding_model.encode(
            self.build_embedding_text(paper),
            normalize_embeddings=True,
        )
        return embedding.tolist()

    def save_paper_info(
        self,
        paper: PaperInfo,
        *,
        overwrite: bool = False,
    ) -> Path:
        filename = self.sanitize_filename(paper.title)
        save_path = self.save_dir / f"{filename}.json"
        if save_path.exists() and not overwrite:
            raise FileExistsError(f"论文已存在：{paper.title}")

        temporary_path = save_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(paper.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf8",
        )
        temporary_path.replace(save_path)
        return save_path

    def parse_pdf(
        self,
        pdf_path: str | Path,
        *,
        save_json: bool = True,
        overwrite: bool = False,
    ) -> PaperInfo:
        pages = self.read_pdf_pages(pdf_path)
        extracted = self.extract_full_paper_info(pages)
        paper = PaperInfo(
            **extracted.model_dump(),
            embedding=self.generate_embedding(extracted),
        )
        if save_json:
            self.save_paper_info(paper, overwrite=overwrite)
        return paper

    def load_paper_info(self, title: str) -> PaperInfo:
        path = self.save_dir / f"{self.sanitize_filename(title)}.json"
        if not path.exists():
            raise FileNotFoundError(f"论文不存在：{title}")
        return PaperInfo.model_validate_json(path.read_text(encoding="utf8"))

    def load_all_papers(self) -> list[PaperInfo]:
        papers: list[PaperInfo] = []
        for path in sorted(self.save_dir.glob("*.json")):
            if path.name == "id_mapping.json":
                continue
            try:
                papers.append(
                    PaperInfo.model_validate_json(
                        path.read_text(encoding="utf8")
                    )
                )
            except (ValueError, json.JSONDecodeError):
                continue
        return papers


if __name__ == "__main__":
    raise SystemExit(
        "请通过 UI 导入 PDF，或在代码中调用 PaperParser.parse_pdf()。"
    )
