import os
import datasets

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.docstore.document  import Document
from  retriever_tool import RetrieverTool
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()
# 读取 HF_TOKEN
hf_token = os.getenv("HF_TOKEN")

# 检查是否成功加载
if hf_token:
    print("HF_TOKEN 已加载:", hf_token)
else:
    print("警告：HF_TOKEN 未找到，请检查 .env 文件是否在当前目录且格式正确。")

# 第一步准备数据
# Load the Hugging Face documentation dataset
knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")

# Filter to include only Transformers documentation
knowledge_base = knowledge_base.filter(lambda row: row["source"].startswith("huggingface/transformers"))

# Convert dataset entries to Document objects with metadata
source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]})
    for doc in knowledge_base
]

# Split documents into smaller chunks for better retrieval
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Characters per chunk
    chunk_overlap=50,  # Overlap between chunks to maintain context
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],  # Priority order for splitting
)
docs_processed = text_splitter.split_documents(source_docs)

print(f"Knowledge base prepared with {len(docs_processed)} document chunks")


# Initialize our retriever tool with the processed documents
retriever_tool = RetrieverTool(docs_processed)

# 创建代理
from smolagents import InferenceClientModel, CodeAgent

# Initialize the agent with our retriever tool
agent = CodeAgent(
    tools=[retriever_tool],  # List of tools available to the agent
    model=InferenceClientModel(api_key=hf_token),  # Default model "Qwen/Qwen3-Next-80B-A3B-Thinking"
    max_steps=4,  # Limit the number of reasoning steps
    verbosity_level=2,  # Show detailed agent reasoning
)

# To use a specific model, you can specify it like this:
# model=InferenceClientModel(model_id="meta-llama/Llama-3.3-70B-Instruct")

# 测试检索效果
# Ask a question that requires retrieving information
question = "For a transformers model training, which is slower, the forward or the backward pass?"

# Run the agent to get an answer
agent_output = agent.run(question)

# Display the final answer
print("\nFinal answer:")
print(agent_output)