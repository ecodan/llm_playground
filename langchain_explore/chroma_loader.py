import csv
import re
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

import chromadb
import tiktoken
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


# load_dotenv(find_dotenv())


class ChromaLoader:

    def __init__(self, chroma_db_path: Path, embedding_model_name: str = 'all-MiniLM-L6-v2') -> None:
        super().__init__()
        self.tokenizer = tiktoken.get_encoding('cl100k_base')

        def tiktoken_len(text):
            tokens = self.tokenizer.encode(
                text,
                disallowed_special=()
            )
            return len(tokens)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=20,
            length_function=tiktoken_len,
            separators=["\n\n", "\n", " ", ""]
        )

        # self.embed = OpenAIEmbeddings(
        #     model=embedding_model_name,
        #     openai_api_key=os.getenv("OPENAI_API_KEY")
        # )
        self.embed = SentenceTransformerEmbeddings(
            model_name=embedding_model_name,
        )

        self.chroma_client: chromadb.PersistentClient = chromadb.PersistentClient(path=chroma_db_path)

    def upsert_embeddings(self, records: List[Dict], index_name: str):
        print(f"upserting {len(records)} records to index {index_name}")
        ids = [str(uuid4()) for _ in range(len(records))]
        text_contents = [x['content'] for x in records]
        embeds = self.embed.embed_documents(text_contents)
        # text_records = [json.dumps(x) for x in records]
        metas = [{"source": x['source']} for x in records]

        collection = self.chroma_client.get_or_create_collection(name=index_name)
        collection.upsert(
            ids=ids,
            embeddings=embeds,
            metadatas=metas,
            documents=text_contents,
        )
        print(f"done with upsert; record count={collection.count()}")

    def load_csv(self, csv_path: Path, url_col: str, content_col: str, index_name: str):
        print(f"loading content from {csv_path}")
        # parse and chunk CSV
        contents = []
        with open(csv_path, "r") as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                metadata = {
                    'source': row[url_col],
                }
                record_content = self.text_splitter.split_text(re.sub('\s+', ' ', row[content_col]))
                record_metadatas = [{
                    "chunk": j, "content": content, **metadata
                } for j, content in enumerate(record_content)]
                contents.extend(record_metadatas)

        self.upsert_embeddings(contents, index_name)


if __name__ == '__main__':
    loader = ChromaLoader("/Users/dcripe/dev/ai/gpt/playground/gpt-search/tmp/chroma.db")
    loader.load_csv(
        csv_path=Path('/Users/dcripe/dev/ai/gpt/playground/gpt-search/data/PhotosHelp.csv'),
        content_col="Articles",
        url_col="links-href",
        index_name="help-content-index"
    )
