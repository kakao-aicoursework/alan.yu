"""Welcome to Reflex! This file outlines the steps to create a basic app."""
from langchain.text_splitter import CharacterTextSplitter
from rxconfig import config

import os
import reflex as rx
import openai
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain import LLMMathChain
from langchain.chat_models import ChatOpenAI
from chatbot import style
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.tools import Tool
from langchain.prompts.chat import ChatPromptTemplate
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from langchain.document_loaders import TextLoader
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

search = GoogleSearchAPIWrapper(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    google_cse_id=os.getenv("GOOGLE_CSE_ID")
)

chat_file = os.path.join("hist.json")

CHROMA_PERSIST_DIR = os.path.join("chroma-persist")
CHROMA_COLLECTION_NAME = "dosu-bot"


def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()

    return prompt_template


def create_chain(llm, template_path, output_key):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=read_prompt_template(template_path)
        ),
        output_key=output_key,
        verbose=True,
    )


llm = ChatOpenAI(temperature=0.1, max_tokens=200, model="gpt-3.5-turbo")

chain1 = create_chain(llm, os.path.join("prompts", "prompt_template1.txt"), "output")
search_value_check_chain = create_chain(llm, os.path.join("prompts", "search_value_check.txt"), "output")
search_compression_chain = create_chain(llm, os.path.join("prompts", "search_compress.txt"), "output")

_db = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=OpenAIEmbeddings(),
    collection_name=CHROMA_COLLECTION_NAME,
)

_retriever = _db.as_retriever()


def query_db(query: str, use_retriever: bool = False) -> list[str]:
    if use_retriever:
        docs = _retriever.get_relevant_documents(query)
    else:
        docs = _db.similarity_search(query)

    str_docs = [doc.page_content for doc in docs]
    return str_docs


search_tool = Tool(
    name="Google Search",
    description="Search Google for recent results.",
    func=search.run,
)

HISTORY_DIR = os.path.join("chat_histories")


def upload_embedding_from_file(file_path):
    documents = TextLoader(file_path).load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(docs, end='\n\n\n')

    Chroma.from_documents(
        docs,
        OpenAIEmbeddings(),
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print(f'{file_path} upload success')


upload_embedding_from_file(os.path.join("datas", "project_data_카카오소셜.txt"))
upload_embedding_from_file(os.path.join("datas", "project_data_카카오싱크.txt"))
upload_embedding_from_file(os.path.join("datas", "project_data_카카오톡채널.txt"))


def load_conversation_history(conversation_id: str):
    file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
    return FileChatMessageHistory(file_path)


def log_user_message(history: FileChatMessageHistory, user_message: str):
    history.add_user_message(user_message)


def log_bot_message(history: FileChatMessageHistory, bot_message: str):
    history.add_ai_message(bot_message)


def get_chat_history(conversation_id: str):
    history = load_conversation_history(conversation_id)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="user_message",
        chat_memory=history,
    )

    return memory.buffer


def query_web_search(user_message: str) -> str:
    context = {"user_message": user_message, "related_web_search_results": search_tool.run(user_message)}

    has_value = search_value_check_chain.run(context)

    print(has_value)
    if has_value == "Y":
        return search_compression_chain.run(context)
    else:
        return ""


def generate_answer(user_message, conversation_id: str = 'fa1010') -> dict[str, str]:
    history_file = load_conversation_history(conversation_id)

    context = dict(user_message=user_message)
    context["input"] = context["user_message"]
    context["chat_history"] = get_chat_history(conversation_id)

    context["related_documents"] = query_db(context["user_message"])

    answer = ""
    for step in [chain1]:
        context = step(context)
        answer += context[step.output_key]
        answer += "\n\n"

    log_user_message(history_file, user_message)
    log_bot_message(history_file, answer)
    return {"answer": answer}


class Message(rx.Model):
    text: str = ""
    created_at: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    me: bool = False

    def __init__(self, text: str, me: bool):
        super().__init__()
        self.text = text
        self.me = me

    def __repr__(self) -> str:
        return "(" + self.text + "-" + str(self.me) + ")"


class State(rx.State):
    """The app state."""
    text: str = ""
    messages: list[Message] = []

    def send(self):
        q = self.text
        print(f"Q: {q}")

        m = Message(q, True)
        self.messages += [m]
        self.text = ""
        yield

        answer = generate_answer(q)
        msg = Message(answer["answer"], False)

        print(f"A: {msg.text}")
        self.messages += [msg]

    pass


def render_message(m: Message):
    return rx.cond(
        m.me,
        rx.box(
            rx.box(m.text,
                   text_align="right",
                   style=style.question_style,
                   margin_y="1em"
                   ),
            text_align="right"
        ),
        rx.box(
            rx.box(m.text,
                   text_align="left",
                   style=style.answer_style,
                   margin_y="1em"
                   )
        )
    )


def chat() -> rx.Component:
    return rx.box(
        rx.foreach(State.messages, lambda msg: render_message(msg)),
        width="100%",
        style=style.frame_style
    )


def action_bar() -> rx.Component:
    return rx.hstack(
        rx.input(placeholder="text to send", on_blur=State.set_text, style=style.input_style),
        rx.button("Send", on_click=State.send, style=style.input_style),
    )


def index() -> rx.Component:
    return rx.container(
        rx.vstack(
            chat(),
            action_bar()
        ),
        center_content=True,
        margin_top="10px"
    )


# Add state and page to the app.
app = rx.App(state=State)
app.add_page(index)
app.compile()
