import json
from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover - optional dependency
    faiss = None


class FaissStore:

    def __init__(
        self,
        index_path="data/faiss.index",
        mapping_path="data/id_mapping.json"
    ):

        self.index_path = Path(
            index_path
        )

        self.mapping_path = Path(
            mapping_path
        )

        self.index = None

        self.id_mapping = {}

    # ==================================
    # 加载索引
    # ==================================

    def load(self):
        if faiss is None:
            raise RuntimeError(
                "FAISS is not installed. Install faiss-cpu or use vector_store.search.PaperSearchEngine."
            )

        if not self.index_path.exists():

            raise FileNotFoundError(
                f"FAISS索引不存在: {self.index_path}"
            )

        self.index = faiss.read_index(
            str(self.index_path)
        )

        with open(
            self.mapping_path,
            "r",
            encoding="utf8"
        ) as f:

            self.id_mapping = json.load(f)

        print(
            f"加载FAISS成功"
        )

        print(
            f"向量数量: {self.index.ntotal}"
        )

        print(
            f"向量维度: {self.index.d}"
        )

    # ==================================
    # 保存索引
    # ==================================

    def save(self):
        if faiss is None:
            raise RuntimeError(
                "FAISS is not installed. Install faiss-cpu or use vector_store.search.PaperSearchEngine."
            )

        if self.index is None:

            raise ValueError(
                "索引为空"
            )

        faiss.write_index(
            self.index,
            str(self.index_path)
        )

        with open(
            self.mapping_path,
            "w",
            encoding="utf8"
        ) as f:

            json.dump(
                self.id_mapping,
                f,
                ensure_ascii=False,
                indent=4
            )

    # ==================================
    # 搜索
    # ==================================

    def search(
        self,
        use_query:str,
        top_k=10
    ):
        from sentence_transformers import SentenceTransformer
        embedding_model=SentenceTransformer("BAAI/bge-small-zh-v1.5")
        query_embedding=embedding_model.encode(use_query)

        if self.index is None:

            raise RuntimeError(
                "请先调用 load()"
            )

        query_embedding = np.array(
            [query_embedding],
            dtype=np.float32
        )

        scores, ids = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, idx in zip(
            scores[0],
            ids[0]
        ):

            if idx == -1:
                continue

            mapping = self.id_mapping.get(
                str(idx)
            )

            results.append(
                {
                    "score": float(score),
                    "index": int(idx),
                    "paper": mapping
                }
            )
            for item in results:

  

                print(
            item["score"]
        )

                print(
            item["paper"]
        )


        return results

    # ==================================
    # 获取论文总数
    # ==================================

    def count(self):

        if self.index is None:

            return 0

        return self.index.ntotal

    # ==================================
    # 获取向量维度
    # ==================================

    def dimension(self):

        if self.index is None:

            return 0

        return self.index.d
def updatesearchpaper(user_q:str,top_k:int):
    from .build_faiss import build_faiss
    build_faiss()
    store = FaissStore()
    store.load()
    store.search(user_q, top_k)
    store.save()



if __name__ == "__main__":
    # from sentence_transformers import SentenceTransformer
    # store = FaissStore()
    # # embedding_model=SentenceTransformer("BAAI/bge-small-zh-v1.5")
    # # query_embedding=embedding_model.encode("Action Recognition")

    # store.load()
    # store.search("Action Recognition", top_k=10)
    updatesearchpaper("Action Recognition", top_k=10)
    # print(store.count())
    # print(store.dimension())
    # query_embedding
    # results = store.search(
    #     query_embedding,
    #     top_k=3
    # )

    # for item in results:

    #     print()

    #     print(
    #         item["score"]
    #     )

    #     print(
    #         item["paper"]
    #     )
