from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from schemas import Paper


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_+\-]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def is_ascii_term(token: str) -> bool:
    return any(char.isascii() and char.isalnum() for char in token)


class PaperSearchEngine:
    """Small local retriever over existing paper JSON files.

    It uses lexical scoring so the main agent workflow remains runnable even
    when sentence-transformers or FAISS are not installed yet.
    """

    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)
        self.papers = self._load_papers()
        self.document_frequency = self._build_document_frequency()

    def _load_papers(self) -> list[Paper]:
        papers: list[Paper] = []
        for path in sorted(self.data_dir.glob("*.json")):
            if path.name == "id_mapping.json":
                continue

            with path.open("r", encoding="utf8") as file:
                data = json.load(file)

            if not isinstance(data, dict) or "title" not in data:
                continue

            paper = Paper.from_dict(data, source_file=str(path))
            if self._is_placeholder_paper(paper, path):
                continue

            papers.append(paper)

        return papers

    def _is_placeholder_paper(self, paper: Paper, path: Path) -> bool:
        title = paper.title.strip().lower()
        text = paper.search_text().lower()
        filename = path.stem.lower()
        placeholder_markers = [
            "论文的题目",
            "璁烘枃鐨勯",
            "latex论文写作模板",
            "latex",
        ]
        if title in {"论文的题目", "paper title"}:
            return True
        if any(marker in filename for marker in placeholder_markers[:2]):
            return True
        return "latex" in text and "模板" in text

    def _build_document_frequency(self) -> Counter[str]:
        document_frequency: Counter[str] = Counter()
        for paper in self.papers:
            document_frequency.update(set(tokenize(paper.search_text())))
        return document_frequency

    def search(self, query: str, top_k: int = 5) -> list[Paper]:
        if not query.strip() or not self.papers:
            return []

        query_tokens = self._expand_query_tokens(query)
        if not query_tokens:
            return []

        scores = []
        for paper in self.papers:
            score = self._score_paper(query_tokens, paper)
            if score > 0:
                scores.append((score, paper))

        scores.sort(key=lambda item: item[0], reverse=True)
        return [
            Paper.from_dict(
                paper.__dict__,
                source_file=paper.source_file,
                score=round(score, 4),
            )
            for score, paper in scores[:top_k]
        ]

    def _expand_query_tokens(self, query: str) -> list[str]:
        tokens = tokenize(query)
        lowered = query.lower()

        expansions: list[str] = []
        phrase_map = {
            "视频": ["video"],
            "长视频": ["long", "video", "hour-long"],
            "视频理解": ["video", "understanding"],
            "流式": ["streaming"],
            "实时": ["real-time", "streaming"],
            "图像": ["image", "vision"],
            "视觉": ["vision", "visual"],
            "分割": ["segmentation"],
            "检测": ["detection"],
            "超分": ["super-resolution"],
            "去雾": ["dehazing"],
            "缺陷": ["defect"],
            "缓存": ["cache", "kv"],
            "记忆": ["memory"],
        }
        for phrase, mapped_tokens in phrase_map.items():
            if phrase in lowered:
                expansions.extend(mapped_tokens)

        return tokens + expansions

    def _score_paper(self, query_tokens: list[str], paper: Paper) -> float:
        text_tokens = tokenize(paper.search_text())
        if not text_tokens:
            return 0.0

        counts = Counter(text_tokens)
        doc_len = len(text_tokens)
        total_docs = max(len(self.papers), 1)
        title = paper.title.lower()
        keywords = " ".join(paper.keywords).lower()

        score = 0.0
        for token in query_tokens:
            tf = counts[token] / doc_len
            if tf == 0:
                continue
            df = self.document_frequency.get(token, 0)
            idf = math.log((1 + total_docs) / (1 + df)) + 1
            weight = 8.0 if is_ascii_term(token) else 0.6
            score += tf * idf * weight

        for token in set(query_tokens):
            if token in title:
                score += 4.0 if is_ascii_term(token) else 0.25
            if token in keywords:
                score += 2.0 if is_ascii_term(token) else 0.15

        return score
