from __future__ import annotations

from langchain_core.output_parsers import PydanticOutputParser

from paper_parser import PaperChunkNotes, PaperInfoTmp, PaperParser


def test_chunk_pages_covers_every_page() -> None:
    pages = [
        "PAGE_ONE_UNIQUE " + "a" * 25,
        "PAGE_TWO_UNIQUE " + "b" * 25,
        "PAGE_THREE_UNIQUE " + "c" * 25,
    ]
    chunks = PaperParser.chunk_pages(pages, max_chars=45)
    combined = "\n".join(chunks)

    assert len(chunks) >= 3
    for marker in [
        "PAGE_ONE_UNIQUE",
        "PAGE_TWO_UNIQUE",
        "PAGE_THREE_UNIQUE",
    ]:
        assert marker in combined


def test_extract_paper_info_processes_all_chunks() -> None:
    parser = PaperParser.__new__(PaperParser)
    parser.chunk_size = 10
    seen_chunks: list[str] = []

    def fake_extract(chunk: str, index: int, count: int) -> PaperChunkNotes:
        seen_chunks.append(chunk)
        return PaperChunkNotes(section_summary=f"{index}/{count}:{chunk}")

    def fake_synthesize(notes: list[PaperChunkNotes]) -> PaperInfoTmp:
        assert len(notes) == 4
        return PaperInfoTmp(
            title="Full Paper",
            authors=[],
            abstract="abstract",
            summary="summary",
            keywords=["full", "text", "test", "paper", "parser"],
            contributions=[],
            limitations=[],
        )

    parser.extract_chunk_notes = fake_extract
    parser.synthesize_paper_info = fake_synthesize

    source = "0123456789ABCDEFGHIJabcdefghijXYZ"
    result = parser.extract_paper_info(source)

    assert "".join(seen_chunks) == source
    assert result.title == "Full Paper"


def test_long_note_sets_are_hierarchically_compacted() -> None:
    parser = PaperParser.__new__(PaperParser)
    merge_calls: list[list[str]] = []

    def fake_merge(
        notes: list[PaperChunkNotes],
        batch_index: int,
        batch_count: int,
    ) -> PaperChunkNotes:
        summaries = [note.section_summary for note in notes]
        merge_calls.append(summaries)
        return PaperChunkNotes(
            section_summary=f"batch {batch_index}/{batch_count}"
        )

    parser.merge_note_batch = fake_merge
    notes = [
        PaperChunkNotes(section_summary=f"section-{index}-" + "x" * 80)
        for index in range(8)
    ]
    compacted = parser.compact_notes(notes, max_payload_chars=500)

    assert merge_calls
    assert len(compacted) < len(notes)
    assert sum(len(batch) for batch in merge_calls) >= len(notes)


def test_structured_parser_accepts_json_inside_markdown() -> None:
    parser = PydanticOutputParser(pydantic_object=PaperChunkNotes)
    content = """
下面是结果：
```json
{
  "section_summary": "介绍 DETR 的端到端目标检测方法",
  "title_candidates": ["End-to-End Object Detection with Transformers"],
  "author_candidates": [],
  "methods": ["set prediction", "transformer encoder-decoder"],
  "results": [],
  "contributions": ["removes hand-designed NMS components"],
  "limitations": [],
  "keywords": ["DETR", "object detection"]
}
```
"""

    result = PaperParser._parse_structured_content(
        content,
        parser,
        PaperChunkNotes,
    )

    assert result.section_summary.startswith("介绍 DETR")
    assert "DETR" in result.keywords


def test_invoke_structured_repairs_invalid_json_once() -> None:
    class FakeResponse:
        def __init__(self, content: str):
            self.content = content

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, prompt: str) -> FakeResponse:
            self.calls += 1
            if self.calls == 1:
                return FakeResponse('{"section_summary": "broken",')
            return FakeResponse(
                '{"section_summary": "repaired",'
                '"title_candidates": [],'
                '"author_candidates": [],'
                '"methods": [],'
                '"results": [],'
                '"contributions": [],'
                '"limitations": [],'
                '"keywords": []}'
            )

    fake_llm = FakeLLM()
    paper_parser = PaperParser.__new__(PaperParser)
    paper_parser.llm = fake_llm

    result = paper_parser.invoke_structured(
        "extract notes",
        PydanticOutputParser(pydantic_object=PaperChunkNotes),
        PaperChunkNotes,
    )

    assert result.section_summary == "repaired"
    assert fake_llm.calls == 2


if __name__ == "__main__":
    test_chunk_pages_covers_every_page()
    test_extract_paper_info_processes_all_chunks()
    test_long_note_sets_are_hierarchically_compacted()
    test_structured_parser_accepts_json_inside_markdown()
    test_invoke_structured_repairs_invalid_json_once()
    print("full text parser tests passed")
