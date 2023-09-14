"""Welcome to Pynecone! This file outlines the steps to create a basic app."""
from pcconfig import config

import pynecone as pc
from pynecone.base import Base
from dataclasses import dataclass
from datetime import date

docs_url = "https://pynecone.io/docs/getting-started/introduction"
filename = f"{config.app_name}/{config.app_name}.py"


@dataclass
class Message:
    text: str
    created_at: date = date.today()
    me: bool = False


class State(pc.State):
    """The app state."""
    text: str = ""
    messages: list[Message] = []

    pass


def message(m: Message):
    print(m)
    # return pc.box(
    #     m.text,
    #     text_align="right" if m.me == True else "left"
    # )


def chat():
    return pc.box(
        "Chat"
    )


def action_bar() -> pc.Component:
    return pc.hstack(
        pc.input(placeholder="Ask a question"),
        pc.button("Ask"),
    )


def index() -> pc.Component:
    return pc.container(
        pc.vstack(
            chat(),
            action_bar()
        ),
        center_content=True,
        bg="lightblue",
        margin_top="10px"
    )


# Add state and page to the app.
app = pc.App(state=State)
app.add_page(index)
app.compile()
