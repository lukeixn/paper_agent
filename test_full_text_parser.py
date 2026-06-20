from __future__ import annotations

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


if __name__ == "__main__":
    test_chunk_pages_covers_every_page()
    test_extract_paper_info_processes_all_chunks()
    test_long_note_sets_are_hierarchically_compacted()
    print("full text parser tests passed")
