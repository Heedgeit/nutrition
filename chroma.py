import os
import chromadb
from chromadb.utils import embedding_functions


class ChromaVectorStore:

    def __init__(self, db_path: str, collection_name: str = "diet_col"):
        """
        Initialize ChromaDB client and embedding function
        """
        self.db_path = db_path
        self.collection_name = collection_name

        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # Make sure the DB folder exists
        os.makedirs(self.db_path, exist_ok=True)

        # Initialize persistent client
        self.client = chromadb.PersistentClient(path=self.db_path)

        # Try to get the collection if it exists; otherwise, create
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        """
        Helper: get the collection if it exists, else create it
        """
        try:
            collection = self.client.get_collection(name=self.collection_name)
            return collection
        except chromadb.errors.CollectionNotFoundError:
            # Collection does not exist, create it
            collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            return collection

    def prepare_documents(self, splits):
        """
        Extract documents, ids and metadata from doc splits
        """
        documents = [doc.page_content for doc in splits]

        ids = [
            f"{doc.metadata['dl_meta']['origin']['filename']}_"
            f"{doc.metadata['dl_meta']['doc_items'][0]['self_ref']}"
            for doc in splits
        ]

        metadatas = []

        for doc in splits:
            filename = doc.metadata["dl_meta"]["origin"]["filename"]
            page = doc.metadata["dl_meta"]["doc_items"][0]["prov"][0]["page_no"]

            metadatas.append({
                "source": filename,
                "page": page
            })

        return documents, metadatas, ids

    def upsert_documents(self, splits):
        """
        Insert documents into ChromaDB
        """
        # Ensure collection exists
        if self.collection is None:
            self.collection = self._get_or_create_collection()

        documents, metadatas, ids = self.prepare_documents(splits)

        self.collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def query(self, query_text, n_results=1):
        """
        Query the vector database
        """
        # Ensure collection exists
        if self.collection is None:
            self.collection = self._get_or_create_collection()

        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )

        return results