# import faiss

# index = faiss.read_index(
#     "data/faiss.index"
# )

# print(
#     "论文数量:",
#     index.ntotal
# )

# print(
#     "向量维度:",
#     index.d
# )
import json
with open("data/id_mapping.json","r",encoding="utf-8") as f:
    id_mapping = json.load(f)
    print(id_mapping["0"]["title"])
print(id_mapping)