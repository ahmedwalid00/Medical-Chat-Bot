from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
# --- Import necessary components for custom retriever ---
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List
from llama_index.core.schema import NodeWithScore
# --- End custom retriever imports ---
from llama_index.core import SimpleDirectoryReader , VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.core import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
import chromadb
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- Define Custom LangChain Retriever ---
class CustomLlamaIndexRetriever(BaseRetriever):
    """Custom LangChain retriever that correctly wraps a LlamaIndex retriever."""
    llama_index_retriever: object # Will hold the object from index.as_retriever()

    # This defines how LangChain should get documents using this retriever
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve relevant documents using the wrapped LlamaIndex retriever."""

        # Use the LlamaIndex retriever's standard 'retrieve' method
        retrieved_nodes: List[NodeWithScore] = self.llama_index_retriever.retrieve(query)

        # Convert LlamaIndex NodeWithScore objects to LangChain Document objects
        langchain_documents = []
        for node_with_score in retrieved_nodes:
            node = node_with_score.node # The actual text node
            text = node.get_content()   # Get the text content
            metadata = node.metadata or {} # Get metadata
            # Optionally add the score to metadata
            metadata["score"] = node_with_score.score
            langchain_documents.append(Document(page_content=text, metadata=metadata))

        return langchain_documents

# --- End Custom LangChain Retriever ---


def create_llama_index_components(file_path, data_dir="D:\\Generative ai\\Medical-ChatBot\\data"):
    """
    Sets up LlamaIndex, creates the index, and returns the core LlamaIndex retriever object.
    """
    documents = SimpleDirectoryReader(data_dir).load_data()

    collection_name = "medical-bot"
    db = chromadb.PersistentClient(path=file_path)
    collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Use updated imports for embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.embed_model = embedding_model
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )

    # Return the LlamaIndex retriever object itself
    llama_index_core_retriever = index.as_retriever(similarity_top_k=3) # Adjust k as needed
    return llama_index_core_retriever


def langchain_pipeline(file_path) :
    """
    Builds the LangChain ConversationalRetrievalChain using the custom retriever.
    """
    # 1. Get the actual LlamaIndex retriever object
    llama_retriever_object = create_llama_index_components(file_path=file_path)

    # 2. Instantiate our custom LangChain retriever wrapper
    custom_langchain_retriever = CustomLlamaIndexRetriever(
        llama_index_retriever=llama_retriever_object
        )

    # 3. Build the rest of the LangChain pipeline
    # Use updated import for Chat model
    final_model = ChatOpenAI(model='gpt-4o-mini', temperature=0.5)

    memory = ConversationBufferWindowMemory(
        k=2,
        memory_key="chat_history", # Standard key for the chain
        return_messages=True       # Recommended for chat models
        )

    # Use our custom_langchain_retriever in the chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=final_model,
        memory=memory,
        retriever=custom_langchain_retriever, # Use the custom one here!
    )
    return qa_chain