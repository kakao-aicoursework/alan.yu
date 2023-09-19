"""Welcome to Pynecone! This file outlines the steps to create a basic app."""
from pcconfig import config

import os.path
import pynecone as pc
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain import LLMMathChain
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI

import os

apiKey = open(os.path.dirname(__file__) + "/../../apikey.txt", "r").read()
os.environ["OPENAI_API_KEY"] = apiKey

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
loader = TextLoader("./project_data_카카오싱크.txt")
document = loader.load()

doc = ""

for d in document:
    doc += d.page_content


prompt = PromptTemplate(
    input_variables=["q"],
    template=doc + "를 참고해서 {q}"
)

chain = LLMChain(llm=llm, prompt=prompt)


class Message(pc.Model):
    text: str = ""
    created_at: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    me: int = 0

    def __init__(self, text: str, me: int):
        super().__init__()
        self.text = text
        self.me = me

    def __repr__(self) -> str:
        return "(" + self.text + "-" + str(self.me) + ")"


class State(pc.State):
    """The app state."""
    text: str = ""
    messages: list[Message] = []

    def send(self):
        q = self.text
        print(f"Q: {q}")

        m = Message(q, 1)
        self.messages += [m]
        self.text = ""
        yield

        answer = chain.run(q)
        print(f"A: {answer}")
        self.messages += [answer]

    pass


def render_message(m: Message):
    return pc.box(m.text, text_align="right" if m.me == 1 else "left", width="100%")


def chat() -> pc.Component:
    return pc.box(
        pc.foreach(State.messages, lambda msg: render_message(msg)),
        width="100%",
    )


def action_bar() -> pc.Component:
    return pc.hstack(
        pc.input(placeholder="text to send", on_blur=State.set_text),
        pc.button("Send", on_click=State.send),
    )


def index() -> pc.Component:
    return pc.container(
        pc.vstack(
            chat(),
            action_bar()
        ),
        center_content=True,
        margin_top="10px"
    )


# Add state and page to the app.
app = pc.App(state=State)
app.add_page(index)
app.compile()
