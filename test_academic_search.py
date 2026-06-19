from __future__ import annotations

import json

import academic_search
from academic_search import AcademicSearchService


class FakeResponse:
    def __init__(self, works):
        self.works = works

    def raise_for_status(self):
        return None

    def json(self):
        return {"results": self.works}


class FakeLLM:
    def invoke(self, prompt: str):
        start = prompt.index("[")
        end = prompt.rindex("]") + 1
        candidates = json.loads(prompt[start:end])
        scores = [
            {"id": item["id"], "score": 100 - item["id"]}
            for item in candidates
        ]
        return type("Response", (), {"content": json.dumps(scores)})()


def make_work(index: int) -> dict:
    return {
        "id": f"https://openalex.org/W{index}",
        "doi": f"https://doi.org/10.1234/{index}",
        "title": f"Paper {index} about video memory",
        "publication_year": 2020 + index % 5,
        "authorships": [
            {"author": {"display_name": f"Author {index}"}}
        ],
        "abstract_inverted_index": {
            "video": [0],
            "memory": [1],
            str(index): [2],
        },
        "best_oa_location": {
            "landing_page_url": f"https://example.org/{index}",
            "pdf_url": f"https://example.org/{index}.pdf",
        },
        "primary_location": {},
        "cited_by_count": index,
        "relevance_score": 100 - index,
    }


def test_search_ranks_one_hundred_candidates() -> None:
    works = [make_work(index) for index in range(100)]
    original_get = academic_search.requests.get
    original_get_llm = academic_search.get_llm
    academic_search.requests.get = (
        lambda *args, **kwargs: FakeResponse(works)
    )
    academic_search.get_llm = lambda **kwargs: FakeLLM()
    try:
        candidates = AcademicSearchService().search(
            "video memory",
            max_candidates=100,
            existing_titles={"Paper 1 about video memory"},
            model_config={"provider": "deepseek", "api_key": "test"},
        )
    finally:
        academic_search.requests.get = original_get
        academic_search.get_llm = original_get_llm

    assert len(candidates) == 100
    assert candidates[0].rank == 1
    assert candidates[0].title == "Paper 0 about video memory"
    assert candidates[1].already_exists
    assert candidates[0].importable


def test_library_import_limit() -> None:
    from paper_library import PaperLibrary

    try:
        PaperLibrary().import_search_candidates(
            [object()] * 11,
            model_config={"provider": "offline"},
        )
    except ValueError as exc:
        assert "10" in str(exc)
    else:
        raise AssertionError("Expected the 10-paper import limit")


if __name__ == "__main__":
    test_search_ranks_one_hundred_candidates()
    test_library_import_limit()
    print("academic search tests passed")
