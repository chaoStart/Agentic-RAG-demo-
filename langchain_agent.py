import bs4
from langchain.agents import create_agent
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from model_utils.embeding_vector import embeddings
from model_utils.private_model import model

# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("https://huggingface.co/docs/smolagents/examples/rag",),
    # bs_kwargs=dict(
    #     parse_only=bs4.SoupStrainer(
    #         class_=("post-content", "post-title", "post-header")
    #     )
    # ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# 方式1 内存向量数据
# Index chunks
vector_store = InMemoryVectorStore(embeddings)
_ = vector_store.add_documents(documents=all_splits)

# 方式2 milvus存储向量数据
from langchain_milvus import Milvus

# vector_store = Milvus(
#     embedding_function=embeddings,
#     connection_args={
#         "host": "localhost",
#         "port": "19530",
#     },
#     index_params={
#         "index_type": "FLAT",
#         "metric_type": "L2"
#     }
# )
# _ = vector_store.add_documents(documents=all_splits)

# Construct a tool for retrieving context
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join((f"Source: {doc.metadata}\nContent: {doc.page_content}") for doc in retrieved_docs)
    return serialized, retrieved_docs

tools = [retrieve_context]
# If desired, specify custom instructions
prompt = (
    "You have access to a tool that retrieves context from a blog post. "
    "Use the tool to help answer user queries."
)
agent = create_agent(model, tools, system_prompt=prompt)

query = (
    "What is the limitations of Traditional RAG?\n\n"
    "Once you get the answer, look up common extensions of that method."
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()