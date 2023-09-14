import pynecone as pc

class PyneconefirstConfig(pc.Config):
    pass

config = PyneconefirstConfig(
    app_name="pynecone_first",
    db_url="sqlite:///pynecone.db",
    env=pc.Env.DEV,
)