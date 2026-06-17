import json
from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover - optional dependency
    faiss = None


PAPER_DIR = Path(
    "data"
)

FAISS_PATH = Path(
    "data/faiss.index"
)

MAPPING_PATH = Path(
    "data/id_mapping.json"
)


def load_papers():

    papers = []

    for file in PAPER_DIR.glob(
        "*.json"
    ):
        if file.name=="id_mapping.json":
            continue

        with open(
            file,
            "r",
            encoding="utf8"
        ) as f:

            paper = json.load(f)

        papers.append(
            paper
        )

    return papers


def build_faiss():
    if faiss is None:
        raise RuntimeError(
            "FAISS is not installed. Install faiss-cpu before rebuilding the FAISS index."
        )

    papers = load_papers()

    if len(papers) == 0:

        raise ValueError(
            "未找到论文数据"
        )

    vectors = []

    id_mapping = {}

    print(
        f"发现 {len(papers)} 篇论文"
    )

    for idx, paper in enumerate(
        papers
    ):

        vectors.append(
            paper["embedding"]
        )

        id_mapping[str(idx)] = {
            "title": paper["title"],
            "file": paper["title"]
        }

    vectors = np.array(
        vectors,
        dtype=np.float32
    )

    print(
        f"向量维度: {vectors.shape}"
    )

    dimension = vectors.shape[1]

    # 因为你用了 normalize_embeddings=True
    # 所以直接使用 Inner Product
    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        vectors
    )

    FAISS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    faiss.write_index(
        index,
        str(FAISS_PATH)
    )

    with open(
        MAPPING_PATH,
        "w",
        encoding="utf8"
    ) as f:

        json.dump(
            id_mapping,
            f,
            ensure_ascii=False,
            indent=4
        )

    print()

    print(
        f"索引建立完成"
    )

    print(
        f"论文数: {index.ntotal}"
    )

    print(
        f"FAISS文件: {FAISS_PATH}"
    )

    print(
        f"映射文件: {MAPPING_PATH}"
    )


if __name__ == "__main__":

    build_faiss()
