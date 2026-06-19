from __future__ import annotations

import ipaddress
import json
import math
import re
import socket
from dataclasses import dataclass
from io import BytesIO
from typing import Any
from urllib.parse import urlparse

import requests

from models.langchain_llm import OfflineLLM, get_llm


OPENALEX_API_URL = "https://api.openalex.org/works"
USER_AGENT = "PaperAgent/1.0 (local research assistant)"
MAX_PDF_BYTES = 50 * 1024 * 1024


@dataclass
class SearchCandidate:
    rank: int
    title: str
    authors: list[str]
    year: int | None
    abstract: str
    doi: str
    landing_page_url: str
    pdf_url: str
    cited_by_count: int
    semantic_score: float
    final_score: float
    already_exists: bool = False

    @property
    def importable(self) -> bool:
        return bool(self.pdf_url) and not self.already_exists


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    if not inverted_index:
        return ""
    positioned_words = [
        (position, word)
        for word, positions in inverted_index.items()
        for position in positions
    ]
    positioned_words.sort(key=lambda item: item[0])
    return " ".join(word for _, word in positioned_words)


def ensure_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("无效的公开 PDF 地址。")

    addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 443)
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        ):
            raise ValueError("拒绝访问非公网 PDF 地址。")


class AcademicSearchService:
    def __init__(
        self,
        timeout: int = 30,
    ):
        self.timeout = timeout

    def search(
        self,
        query: str,
        *,
        max_candidates: int = 100,
        existing_titles: set[str] | None = None,
        model_config: dict[str, Any] | None = None,
    ) -> list[SearchCandidate]:
        query = query.strip()
        if not query:
            raise ValueError("检索内容不能为空。")

        response = requests.get(
            OPENALEX_API_URL,
            params={
                "search": query,
                "filter": "is_oa:true",
                "per-page": min(max_candidates, 100),
                "select": (
                    "id,doi,title,publication_year,authorships,"
                    "abstract_inverted_index,best_oa_location,"
                    "primary_location,cited_by_count,relevance_score"
                ),
            },
            headers={"User-Agent": USER_AGENT},
            timeout=self.timeout,
        )
        response.raise_for_status()
        works = response.json().get("results", [])[:max_candidates]
        if not works:
            return []

        records = [self._work_to_record(work) for work in works]
        relevance_scores = [
            float(record["relevance_score"] or 0.0) for record in records
        ]
        relevance_scores = self._normalize(relevance_scores)
        citation_scores = [
            math.log1p(record["cited_by_count"]) for record in records
        ]
        citation_scores = self._normalize(citation_scores)
        lexical_scores = [
            self._lexical_score(query, record) for record in records
        ]
        base_scores = [
            relevance * 0.70 + lexical * 0.25 + citation * 0.05
            for relevance, lexical, citation in zip(
                relevance_scores,
                lexical_scores,
                citation_scores,
            )
        ]
        ai_scores = self._ai_scores(query, records, model_config)
        final_scores = [
            ai_score * 0.75 + base_score * 0.25
            for ai_score, base_score in zip(ai_scores, base_scores)
        ]
        existing_titles = {
            title.strip().casefold() for title in (existing_titles or set())
        }

        ordered = sorted(
            zip(records, ai_scores, final_scores),
            key=lambda item: item[2],
            reverse=True,
        )
        return [
            SearchCandidate(
                rank=index,
                title=record["title"],
                authors=record["authors"],
                year=record["year"],
                abstract=record["abstract"],
                doi=record["doi"],
                landing_page_url=record["landing_page_url"],
                pdf_url=record["pdf_url"],
                cited_by_count=record["cited_by_count"],
                semantic_score=round(float(semantic_score), 4),
                final_score=round(float(final_score), 4),
                already_exists=record["title"].strip().casefold()
                in existing_titles,
            )
            for index, (record, semantic_score, final_score) in enumerate(
                ordered,
                start=1,
            )
        ]

    def _ai_scores(
        self,
        query: str,
        records: list[dict[str, Any]],
        model_config: dict[str, Any] | None,
    ) -> list[float]:
        model_config = model_config or {}
        llm = get_llm(
            provider=model_config.get("provider"),
            api_key=model_config.get("api_key"),
            model_name=model_config.get("model_name"),
            base_url=model_config.get("base_url"),
            temperature=0.0,
        )
        if isinstance(llm, OfflineLLM):
            raise RuntimeError("AI 排名需要配置 DeepSeek 或 OpenAI API Key。")

        candidate_payload = [
            {
                "id": index,
                "title": record["title"],
                "year": record["year"],
                "abstract": record["abstract"][:260],
            }
            for index, record in enumerate(records)
        ]
        prompt = f"""
你是学术检索排序器。根据用户研究问题，为下面每一篇候选论文评估主题契合度。

用户问题：
{query}

候选论文：
{json.dumps(candidate_payload, ensure_ascii=False)}

请只返回 JSON 数组，不要解释。每项格式：
{{"id": 整数, "score": 0到100之间的数字}}

必须为所有候选返回分数。关注研究主题和方法契合度，不要仅按引用量排序。
"""
        response = llm.invoke(prompt).content.strip()
        response = re.sub(r"^```(?:json)?\s*|\s*```$", "", response)
        try:
            scored_items = json.loads(response)
        except json.JSONDecodeError as exc:
            raise RuntimeError("AI 排名返回格式无法解析。") from exc

        score_map = {
            int(item["id"]): max(0.0, min(float(item["score"]) / 100.0, 1.0))
            for item in scored_items
            if isinstance(item, dict) and "id" in item and "score" in item
        }
        if len(score_map) < max(1, len(records) // 2):
            raise RuntimeError("AI 排名返回的候选数量不足。")
        return [score_map.get(index, 0.0) for index in range(len(records))]

    @staticmethod
    def _normalize(values: list[float]) -> list[float]:
        if len(values) == 0:
            return values
        minimum = min(values)
        maximum = max(values)
        if maximum <= minimum:
            return [0.0 for _ in values]
        return [
            (value - minimum) / (maximum - minimum) for value in values
        ]

    @staticmethod
    def _lexical_score(query: str, record: dict[str, Any]) -> float:
        query_tokens = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", query.lower()))
        text = f"{record['title']} {record['abstract']}".lower()
        if not query_tokens:
            return 0.0
        matches = sum(1 for token in query_tokens if token in text)
        return matches / len(query_tokens)

    @staticmethod
    def _work_to_record(work: dict[str, Any]) -> dict[str, Any]:
        best_location = work.get("best_oa_location") or {}
        primary_location = work.get("primary_location") or {}
        authors = [
            authorship.get("author", {}).get("display_name", "")
            for authorship in work.get("authorships") or []
        ]
        authors = [author for author in authors if author]
        return {
            "title": work.get("title") or "Untitled",
            "authors": authors,
            "year": work.get("publication_year"),
            "abstract": reconstruct_abstract(
                work.get("abstract_inverted_index")
            ),
            "doi": work.get("doi") or "",
            "landing_page_url": (
                best_location.get("landing_page_url")
                or primary_location.get("landing_page_url")
                or work.get("id")
                or ""
            ),
            "pdf_url": best_location.get("pdf_url") or "",
            "cited_by_count": int(work.get("cited_by_count") or 0),
            "relevance_score": work.get("relevance_score") or 0.0,
        }

    def download_pdf(self, candidate: SearchCandidate) -> BytesIO:
        if not candidate.pdf_url:
            raise ValueError("该论文没有公开 PDF。")
        ensure_public_url(candidate.pdf_url)

        with requests.get(
            candidate.pdf_url,
            headers={"User-Agent": USER_AGENT},
            timeout=self.timeout,
            stream=True,
            allow_redirects=True,
        ) as response:
            response.raise_for_status()
            ensure_public_url(response.url)
            content_length = int(response.headers.get("content-length") or 0)
            if content_length > MAX_PDF_BYTES:
                raise ValueError("PDF 超过 50 MB，已跳过。")

            buffer = BytesIO()
            for chunk in response.iter_content(64 * 1024):
                if not chunk:
                    continue
                buffer.write(chunk)
                if buffer.tell() > MAX_PDF_BYTES:
                    raise ValueError("PDF 超过 50 MB，已跳过。")

        buffer.seek(0)
        if buffer.read(5) != b"%PDF-":
            raise ValueError("下载内容不是有效 PDF。")
        buffer.seek(0)
        buffer.name = f"{candidate.title}.pdf"
        return buffer
