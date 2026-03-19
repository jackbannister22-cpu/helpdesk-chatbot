from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

print("Loading Q&A knowledge files...")

# Load all Q&A files
documents = SimpleDirectoryReader("qa_blocks").load_data()

print("Splitting knowledge into smaller chunks...")

splitter = SentenceSplitter(
    chunk_size=400,
    chunk_overlap=50
)

nodes = splitter.get_nodes_from_documents(documents)

print("Configuring AI models...")

Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-large"
)

Settings.llm = OpenAI(
    model="gpt-4.1"
)

print("Building searchable knowledge index...")

index = VectorStoreIndex(nodes)

query_engine = index.as_query_engine(
    similarity_top_k=5
)

print("\nKnowledge chatbot ready.")
print("Ask a question. Type 'exit' to stop.\n")

while True:
    question = input("Question: ")

    if question.lower() in ["exit", "quit"]:
        print("Chatbot stopped.")
        break

    response = query_engine.query(question)

    print("\nAnswer:\n")
    print(response)
    print("\n")