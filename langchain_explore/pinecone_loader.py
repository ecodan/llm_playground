import csv
from pathlib import Path
from typing import List, Dict
import os
import re
from uuid import uuid4

import pinecone
import tiktoken
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone import Index

load_dotenv(find_dotenv())


class PineconeLoader:

    def __init__(self, embedding_model_name:str = 'text-embedding-ada-002') -> None:
        super().__init__()
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENV")
        )
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

        self.embed = OpenAIEmbeddings(
            model=embedding_model_name,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    def upsert_embeddings(self, records:List[Dict], index_name:str):
        ids = [str(uuid4()) for _ in range(len(records))]
        text_contents = [x['content'] for x in records]
        embeds = self.embed.embed_documents(text_contents)
        # create index (if needed)
        if index_name not in pinecone.list_indexes():
            # we create a new index
            pinecone.create_index(
                name=index_name,
                metric='cosine',
                dimension=len(embeds[0])  # 1536 dim of text-embedding-ada-002
            )
        index = Index(index_name)

        index.upsert(vectors=zip(ids, embeds, records))



    def load_web_pages(self, urls:List, index:str):
        print(f"loading content from {len(urls)} pages")
        # create index (if needed)

        # scrape data
        # loader = WebBaseLoader(urls)
        # scrape_data = loader.load()
        # print(scrape_data)

        # bs4_docs = loader.scrape_all(urls)
        # for doc in bs4_docs:
        #     print(f"article: {doc.find_all('help-content')}")

        # chunk text

        # load to DB
        raise NotImplemented()

    def load_csv(self, csv_path:Path, url_col:str, content_col:str, index_name:str):
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
    loader = PineconeLoader()
    loader.load_csv(
        csv_path=Path('/Users/dcripe/dev/ai/gpt/playground/gpt-search/data/PhotosHelp.csv'),
        content_col="Articles",
        url_col="links-href",
        index_name="help-content-index"
    )
    # loader.load_web_pages(
    #     [
    #         "https://www.amazon.com/gp/help/customer/display.html?nodeId=G202094310&ref_=hp_GDMHBWK3M5RAV5TY_Notice-to-Illinois-Residents",
    #         "https://www.amazon.com/gp/help/customer/display.html?nodeId=G201376540&ref_=hp_GDMHBWK3M5RAV5TY_Amazon-Photos-Terms-of-Use",
    #     ],
    #     "help-pages"
    # )

    # loader = WebBaseLoader([])
    # bs4_docs = loader.scrape_all( [
    #     "https://www.amazon.com/gp/help/customer/display.html?nodeId=G202094310&ref_=hp_GDMHBWK3M5RAV5TY_Notice-to-Illinois-Residents",
    #     "https://www.amazon.com/gp/help/customer/display.html?nodeId=G201376540&ref_=hp_GDMHBWK3M5RAV5TY_Amazon-Photos-Terms-of-Use",
    # ])
    # for doc in bs4_docs:
    #     print(f"article: {doc.find_all('help-content')}")
