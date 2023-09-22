import reflex as rx


class ChatbotConfig(rx.Config):
    pass


config = ChatbotConfig(
    app_name="chatbot",
    db_url="sqlite:///reflex.db",
    env=rx.Env.DEV,
)