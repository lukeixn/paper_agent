import json
import re
from pathlib import Path

import fitz

from pydantic import BaseModel

from langchain_core.output_parsers import (
    PydanticOutputParser
)

from sentence_transformers import SentenceTransformer

from models.langchain_llm import get_llm
from tkinter import Tk
from tkinter.filedialog import (
    askopenfilename,
    askdirectory
)

# ==================================
# Schema
# ==================================

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




# ==================================
# Parser
# ==================================

class PaperParser:

    def __init__(
        self,
        embedding_model_name="BAAI/bge-small-zh-v1.5",
        save_dir="data"
    ):

        self.llm = get_llm()

        self.embedding_model = SentenceTransformer(
            embedding_model_name
        )

        self.output_parser = PydanticOutputParser(
            pydantic_object=PaperInfoTmp
        )

        self.save_dir = Path(save_dir)

        self.save_dir.mkdir(
            parents=True,
            exist_ok=True
        )
        # ==================================
    # 选择单个PDF
    # ==================================

    def select_pdf_file(
        self
    ) -> str:

        root = Tk()

        root.withdraw()

        pdf_path = askopenfilename(

            title="选择论文PDF",

            filetypes=[
                ("PDF文件", "*.pdf")
            ]
        )

        root.destroy()

        return pdf_path

    # ==================================
    # 选择论文目录
    # ==================================

    def select_pdf_directory(
        self
    ) -> str:

        root = Tk()

        root.withdraw()

        directory = askdirectory(

            title="选择论文目录"
        )

        root.destroy()

        return directory

    # ==================================
    # 文件选择器解析单篇论文
    # ==================================

    def parse_pdf_from_dialog(
        self,
        save_json: bool = True
    ) -> PaperInfo | None:

        pdf_path = self.select_pdf_file()

        if not pdf_path:

            print(
                "未选择PDF文件"
            )

            return None

        print(
            f"\n已选择文件:\n{pdf_path}"
        )

        return self.parse_pdf(
            pdf_path,
            save_json
        )

    # ==================================
    # 文件选择器解析整个目录
    # ==================================

    def parse_directory_from_dialog(
        self,
        save_json: bool = True
    ) -> list[PaperInfo]:
        

        directory = self.select_pdf_directory()

        if not directory:

            print(
                "未选择目录"
            )

            return []

        print(
            f"\n已选择目录:\n{directory}"
        )

        return self.parse_directory(
            directory,
            save_json
        )

    # ==================================
    # 文件名清洗
    # ==================================

    def sanitize_filename(
        self,
        filename: str
    ) -> str:

        filename = re.sub(
            r'[<>:"/\\|?*]',
            '_',
            filename
        )

        filename = filename.strip()

        return filename

    # ==================================
    # PDF读取
    # ==================================

    def read_pdf(
        self,
        pdf_path: str
    ) -> str:

        doc = fitz.open(pdf_path)

        text = ""

        for page in doc:

            text += page.get_text()

        return text

    # ==================================
    # 构造Embedding文本
    # ==================================

    def build_embedding_text(
        self,
        paper: PaperInfoTmp
    ) -> str:

        return f"""
标题:
{paper.title}

摘要:
{paper.abstract}

总结:
{paper.summary}

关键词:
{' '.join(paper.keywords)}

主要贡献:
{' '.join(paper.contributions)}
"""

    # ==================================
    # 提取论文信息
    # ==================================

    def extract_paper_info(
        self,
        text: str
    ) -> PaperInfoTmp:

        prompt = f"""
你是一名资深科研助手。

请分析论文。

返回格式必须符合要求。

{self.output_parser.get_format_instructions()}

要求：

1. summary使用中文

2. contributions使用中文

3. limitations使用中文

4. keywords返回5~10个

论文内容：

{text[:25000]}
"""

        response = self.llm.invoke(
            prompt
        )

        print("\n============== LLM原始输出 ==============\n")

        print(response.content)

        paper_info = self.output_parser.parse(
            response.content
        )

        return paper_info

    # ==================================
    # Embedding生成
    # ==================================

    def generate_embedding(
        self,
        paper: PaperInfoTmp
    ):

        embedding_text = self.build_embedding_text(
            paper
        )

        embedding = self.embedding_model.encode(
            embedding_text,
            normalize_embeddings=True
        )

        return embedding

    # ==================================
    # 保存JSON
    # ==================================

    def save_paper_info(
        self,
        paper: PaperInfo
    ):

        filename = self.sanitize_filename(
            paper.title
        )

        save_path = self.save_dir / f"{filename}.json"

        with open(
            save_path,
            "w",
            encoding="utf8"
        ) as f:

            json.dump(
                paper.model_dump(),
                f,
                ensure_ascii=False,
                indent=4
            )

        print(
            f"\n[保存成功] {save_path}"
        )

    # ==================================
    # 加载JSON
    # ==================================

    def load_paper_info(
        self,
        title: str
    ) -> PaperInfo:

        filename = self.sanitize_filename(
            title
        )

        file_path = self.save_dir / f"{filename}.json"

        if not file_path.exists():

            raise FileNotFoundError(
                f"论文不存在: {file_path}"
            )

        with open(
            file_path,
            "r",
            encoding="utf8"
        ) as f:

            data = json.load(f)

        return PaperInfo(
            **data
        )

    # ==================================
    # 获取所有论文
    # ==================================

    def load_all_papers(
        self
    ) -> list[PaperInfo]:

        papers = []

        for file in self.save_dir.glob(
            "*.json"
        ):

            with open(
                file,
                "r",
                encoding="utf8"
            ) as f:

                data = json.load(f)

            papers.append(
                PaperInfo(
                    **data
                )
            )

        return papers

    # ==================================
    # 完整流程
    # ==================================
    # ==================================
# 批量解析目录
# ==================================

    def parse_directory(
        self,
        pdf_dir: str,
        save_json: bool = True
    ) -> list[PaperInfo]:

        pdf_dir = Path(pdf_dir)

        if not pdf_dir.exists():

            raise FileNotFoundError(
                f"目录不存在: {pdf_dir}"
            )

        pdf_files = list(
            pdf_dir.glob("*.pdf")
        )

        print(
            f"\n发现 {len(pdf_files)} 篇PDF论文"
        )

        papers = []

        for idx, pdf_file in enumerate(
            pdf_files,
            start=1
        ):

            print("\n")
            print("=" * 80)

            print(
                f"[{idx}/{len(pdf_files)}]"
            )

            print(
                f"正在处理: {pdf_file.name}"
            )

            try:

                paper = self.parse_pdf(
                    str(pdf_file),
                    save_json=save_json
                )

                papers.append(
                    paper
                )

                print(
                    f"✓ 成功: {paper.title}"
                )

            except Exception as e:

                print(
                    f"✗ 失败: {pdf_file.name}"
                )

                print(
                    f"错误信息: {e}"
                )

        print("\n")
        print("=" * 80)

        print(
            f"处理完成"
        )

        print(
            f"成功: {len(papers)}"
        )

        print(
            f"失败: {len(pdf_files)-len(papers)}"
        )

        return papers
        def is_processed(
        self,
        pdf_path: str
    ) -> bool:
            pdf_path = Path(pdf_path)

            filename = self.sanitize_filename(
                pdf_path.stem
            )

            json_path = (
                self.save_dir /
                f"{filename}.json"
            )

            return json_path.exists()
    def parse_pdf(
        self,
        pdf_path: str,
        save_json: bool = True
    ) -> PaperInfo:

        print(
            f"\n开始解析PDF: {pdf_path}"
        )

        text = self.read_pdf(
            pdf_path
        )

        paper = self.extract_paper_info(
            text
        )

        embedding = self.generate_embedding(
            paper
        )

        paper_info = PaperInfo(

            title=paper.title,

            authors=paper.authors,

            abstract=paper.abstract,

            summary=paper.summary,

            keywords=paper.keywords,

            contributions=paper.contributions,

            limitations=paper.limitations,

            embedding=embedding.tolist()
        )

        if save_json:

            self.save_paper_info(
                paper_info
            )

        return paper_info
if __name__ == "__main__":

    parser = PaperParser()

    # paper = parser.parse_pdf(
    #     "papers/DEA-Net.pdf"
    # )
    # paper=parser.parse_directory(
    #     "papers"
    # )
    parser.parse_pdf_from_dialog(
        save_json=True
    )


    # print("\n")
    # print("=" * 60)

    # print("标题:")
    # print(paper.title)

    # print("\n作者:")
    # print(paper.authors)

    # print("\n关键词:")
    # print(paper.keywords)

    # print("\n主要贡献:")

    # for item in paper.contributions:

    #     print("-", item)

    # print("\n局限性:")

    # for item in paper.limitations:

    #     print("-", item)

    # print("\nEmbedding维度:")

    # print(
    #     len(paper.embedding)
    # )