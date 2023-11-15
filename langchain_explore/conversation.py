import abc
from abc import ABCMeta

import chromadb
from dotenv import load_dotenv, find_dotenv
from langchain.chains import LLMChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate, PromptTemplate
from langchain.schema.vectorstore import VectorStore
from langchain.vectorstores import Chroma

MEMORY_KEY = "chat_memory"


class ChatBot(metaclass=ABCMeta):

    @abc.abstractmethod
    def process_input(self, user_input: str) -> str:
        raise NotImplemented()

    def run(self):
        print("running...")
        while True:
            user_input = input("> ")
            if user_input == "quit":
                print("shutting down...")
                return
            output = self.process_input(user_input)
            print(output)


class BasicChatBot(ChatBot):

    def __init__(self, style: str = None) -> None:
        super().__init__()

        # self.llm = ChatOpenAI(temperature=0)
        self.llm = Ollama(model="vicuna")
        self.tools = []

        sys_prompt = "You are very powerful general purpose assistant. "
        if style is not None:
            sys_prompt += f"You always respond in the style of {style}."
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(sys_prompt),
            MessagesPlaceholder(variable_name=MEMORY_KEY),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        self.memory = ConversationBufferMemory(memory_key=MEMORY_KEY, return_messages=True)
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            verbose=True,
            memory=self.memory
        )

    def process_input(self, user_input: str) -> str:
        result = self.chain({"input": user_input})
        return result['text']


class QAChatBot(ChatBot):

    def __init__(self, vector_store: VectorStore, style: str = None) -> None:
        super().__init__()

        self.llm = ChatOpenAI(temperature=0)
        # self.llm = Ollama(model="vicuna")

        self.vectorstore = vector_store

        chain_type_kwargs = {
            # "verbose":True,
            "memory": ConversationBufferMemory(
                memory_key="history",
                input_key="question"),
        }
        if style is not None:
            prompt_template = "You are a help center agent bot. Use only the following context to answer the question at the end. If you don't know the answer, just say that you don't know. Don't make up an answer. "
            prompt_template += f"Respond in the style of {style}. "
            prompt_template += """
               
                {context}
                
                Question: {question}
                
                Answer: """

            ptplt = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )
            chain_type_kwargs['prompt'] = ptplt

        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 3,
                    "minSimilarityScore": 0.9,
                }
            ),
            chain_type_kwargs=chain_type_kwargs,
        )

    def process_input(self, user_input: str) -> str:
        result = self.qa.run(user_input)
        return result


if __name__ == '__main__':
    load_dotenv(find_dotenv())

    # bot = BasicChatBot(style="shakespearian rhymed couplets")
    # bot = QAChatBot(index_name="help-content-index", style="shakespearian rhymed couplets")

    # create embedding
    # OpenAI
    # model_name = 'text-embedding-ada-002'
    # embed = OpenAIEmbeddings(
    #     model=model_name,
    #     openai_api_key=os.getenv('OPENAI_API_KEY')
    # )
    # SentenceTransformers
    embed = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2",
    )

    # if pinecone
    # pinecone.init(
    #     api_key=os.getenv("PINECONE_API_KEY"),
    #     environment=os.getenv("PINECONE_ENV")
    # )
    # index = pinecone.Index("help-content-index")
    # vector_store = Pinecone(
    #     index, embed.embed_query, "content"
    # )

    # if chroma
    chroma_client: chromadb.PersistentClient = chromadb.PersistentClient(
        path="/Users/dcripe/dev/ai/gpt/playground/gpt-search/tmp/chroma.db")

    vector_store = Chroma(
        collection_name="help-content-index",
        embedding_function=embed,
        persist_directory="/Users/dcripe/dev/ai/gpt/playground/gpt-search/tmp/",
        client=chroma_client,
    )

    bot = QAChatBot(
        vector_store=vector_store,
        style="shakespearean rhymed couplets",
    )
    bot.run()
