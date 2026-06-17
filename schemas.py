from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Paper:
    title: str
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    summary: str = ""
    keywords: list[str] = field(default_factory=list)
    contributions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)
    source_file: str = ""
    score: float = 0.0

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        source_file: str = "",
        score: float = 0.0,
    ) -> "Paper":
        return cls(
            title=str(data.get("title", "")),
            authors=list(data.get("authors") or []),
            abstract=str(data.get("abstract", "")),
            summary=str(data.get("summary", "")),
            keywords=list(data.get("keywords") or []),
            contributions=list(data.get("contributions") or []),
            limitations=list(data.get("limitations") or []),
            embedding=list(data.get("embedding") or []),
            source_file=source_file,
            score=score,
        )

    def search_text(self) -> str:
        parts = [
            self.title,
            self.abstract,
            self.summary,
            " ".join(self.keywords),
            " ".join(self.contributions),
            " ".join(self.limitations),
        ]
        return "\n".join(part for part in parts if part)
