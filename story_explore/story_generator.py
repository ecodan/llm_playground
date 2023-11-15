import json
from pathlib import Path
from typing import List

from dotenv import load_dotenv, find_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import SystemMessagePromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, \
    ChatPromptTemplate

from langchain_explore.conversation import ChatBot
from mm_explore.image_meta import KEY_FNAME, KEY_CAPTION, KEY_PEOPLE, KEY_LON, KEY_LAT, KEY_DATETIME

MEMORY_KEY = "chat_memory"

class StoryBot(ChatBot):

    def __init__(self, meta_files:List[Path], style:str = None) -> None:
        super().__init__()
        self.llm = ChatOpenAI(model_name='gpt-4', temperature=0)
        # self.llm = Ollama(model="vicuna")
        self.tools = []

        sys_prompt = "You are an assistant that curates photo albums. "
        sys_prompt += "You'll be given a list of photos in the format file name, date, gps location, people, objects, and short decription (each field separated with a | character). "
        sys_prompt += "Select the best photos for a photo album based on the input provided. "
        sys_prompt += "You'll need to know (1) how many photos, (2) which people to include, and (3) what mood is desired. Prompt the user for those inputs if not provided. "
        sys_prompt += "As the final output, generate a list of file names in order of optimal display. Only use the file names provided. "
        sys_prompt += "Photos:\n"
        for f in meta_files:
            sys_prompt += StoryBot.meta_to_prompt_string(f)

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

    @classmethod
    def meta_to_prompt_string(cls, meta_file:Path):
        if meta_file.exists():
            with open(meta_file, 'r') as metafile:
                node = json.load(metafile)
                return f"{node[KEY_FNAME]}|{node[KEY_DATETIME]}|{node[KEY_LAT]}, {node[KEY_LON]}|{node[KEY_PEOPLE]}||{node[KEY_CAPTION]}\n"
        return "\n"

    def process_input(self, user_input: str) -> str:
        result = self.chain({"input": user_input})
        return result['text']


if __name__ == '__main__':
    load_dotenv(find_dotenv())

    # get metafiles from image dir
    img_dir = Path("/Users/dcripe/Pictures/ai/semantic/2022/20220718 - europe:africa/06 Sudtirol")
    meta_files = []
    for f in img_dir.iterdir():
        if f.name.endswith(".meta.json"):
            meta_files.append(f)

    bot = StoryBot(meta_files[0:10])
    print(bot.prompt)
    bot.run()
