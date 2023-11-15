from abc import abstractmethod
from typing import Dict

from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore
import chromadb


class MemoryVS(VectorStore):

    def __init__(self, embeddings:Embeddings) -> None:
        super().__init__()
        self.store:Dict = {}

    @abstractmethod
    def load(self):
        raise NotImplementedError()

