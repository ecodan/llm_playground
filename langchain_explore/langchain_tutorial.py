import os
import pinecone
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.llms.openai import OpenAI


load_dotenv(find_dotenv())

if __name__ == '__main__':

    # BASICS
    llm = OpenAI(model_name="text-davinci-003")
    # res = llm("hi there! how are you?")
    # print(res)

    # CHAT AND PROMPTS
    # chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
    # messages = [
    #     SystemMessage(content="you are an expert at photography"),
    #     HumanMessage(content="tell me how to compose a photo in 100 words or less")
    # ]
    # print(f"sending message: {messages}")
    # res = chat(messages)
    # print(f"result {res}")

    template = """
    You are an expert in photography. Explain the concept of {concept} in a couple of lines.
    """
    prompt = PromptTemplate(
        input_variables=["concept"],
        template=template
    )
    # print(f'prompt {prompt.format(concept="f-stop")}')
    # res = llm(prompt.format(concept="f-stop"))
    # print(f"result {res}")

    # CHAINS
    # chain = LLMChain(llm=llm, prompt=prompt)
    # print(f"sending prompt {prompt.format(concept='ISO speed')}")
    # print(f"chain result = {chain.run('ISO speed')}")
    #
    # prompt2 = PromptTemplate(
    #     input_variables=["photo_concept"],
    #     template="take this concept description {photo_concept} and explain it to a kindergartener in 500 words"
    # )
    # chain2 = LLMChain(llm=llm, prompt=prompt2)
    # pchain = SimpleSequentialChain(chains=[chain, chain2], verbose=True)
    # res = pchain.run('focal length')
    # print(f"chains {res}")
    #
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=5)
    # texts = text_splitter.create_documents([res])
    #
    # embeddings = OpenAIEmbeddings(model_name="ada")

    # pinecone.init(
    #     api_key=os.getenv("PINECONE_API_KEY"),
    #     environment=os.getenv("PINECONE_ENV")
    # )
    # index_name = "langchain-quickstart"
    # search = Pinecone.from_documents(texts, embeddings, index_name=index_name)
    #
    # query = "what's important about ISO?"
    # result = search.similarity_search(query)
    # print(f"search result: {result}")

    agent_executor = create_python_agent(
        llm=OpenAI(temperature=0, max_tokens=1000),
        tool = PythonREPLTool(),
        verbose=True
    )
    agent_executor.run("Find the roots (zeros) of the quadratic function 3 * x**2 + 2*x -1")
