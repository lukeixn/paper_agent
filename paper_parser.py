from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import fitz
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from models.langchain_llm import OfflineLLM, get_llm


class PaperInfo(BaseModel):
    title: str
    authors: list[str]
    abstract: str
    summary: str
    keywords: list[str]
    contributions: list[str]
    limitations: list[str]
    embedding: list[float]


class PaperInfoTmp(BaseModel):
    title: str
    authors: list[str]
    abstract: str
    summary: str
    keywords: list[str]
    contributions: list[str]
    limitations: list[str]


class PaperParser:
    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-small-zh-v1.5",
        save_dir: str | Path = "data",
        model_config: dict[str, Any] | None = None,
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

        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.output_parser = PydanticOutputParser(pydantic_object=PaperInfoTmp)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        filename = re.sub(r'[<>:"/\\|?*]', "_", filename).strip().rstrip(".")
        return filename or "untitled-paper"

    @staticmethod
    def read_pdf(pdf_path: str | Path) -> str:
        path = Path(pdf_path)
        with fitz.open(path) as document:
            text = "\n".join(page.get_text() for page in document)

        if not text.strip():
            raise ValueError("PDF 中没有提取到文本，可能是扫描件或受保护文档。")
        return text

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

    def extract_paper_info(self, text: str) -> PaperInfoTmp:
        prompt = f"""
你是一名资深科研助理。请分析下面的论文文本，并严格按照指定 JSON
格式返回论文信息。

{self.output_parser.get_format_instructions()}

要求：
1. summary、contributions 和 limitations 使用中文。
2. keywords 返回 5 到 10 个关键词。
3. 不要虚构论文中没有出现的实验结论。
4. summary 应包含研究问题、核心方法和主要结果。

论文内容：
{text[:30000]}
"""
        response = self.llm.invoke(prompt)
        return self.output_parser.parse(response.content)

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
        text = self.read_pdf(pdf_path)
        extracted = self.extract_paper_info(text)
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
                    PaperInfo.model_validate_json(path.read_text(encoding="utf8"))
                )
            except (ValueError, json.JSONDecodeError):
                continue
        return papers


if __name__ == "__main__":
    raise SystemExit("请通过 UI 上传 PDF，或在代码中调用 PaperParser.parse_pdf()。")
