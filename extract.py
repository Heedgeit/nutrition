from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker
from langchain_docling import DoclingLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter


class DocProcessor:
    def __init__(
        self,
        file_path,
        embed_model_id="sentence-transformers/all-MiniLM-L6-v2",
        export_type=ExportType.DOC_CHUNKS
    ):
        self.file_path = file_path
        self.embed_model_id = embed_model_id
        self.export_type = export_type

        self.loader = DoclingLoader(
            file_path=self.file_path,
            export_type=self.export_type,
            chunker=HybridChunker(tokenizer=self.embed_model_id),
        )

    def load_documents(self):
        """Load documents using Docling."""
        return self.loader.load()

    def process_documents(self):
        """Process documents based on export type."""
        docs = self.load_documents()

        if self.export_type == ExportType.DOC_CHUNKS:
            splits = docs

        elif self.export_type == ExportType.MARKDOWN:
            splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "Header_1"),
                    ("##", "Header_2"),
                    ("###", "Header_3"),
                ]
            )

            splits = []
            chunk_id = 0

            for doc in docs:
                chunks = splitter.split_text(doc.page_content)

                for chunk in chunks:
                    splits.append({
                        "id": chunk_id,
                        "content": chunk,
                        "metadata": doc.metadata
                    })
                    chunk_id += 1

        else:
            raise ValueError(f"Unexpected export type: {self.export_type}")

        return splits


