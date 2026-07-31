from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# Store all loaded documents
documents = []

# Path to data folder
data_folder = "data"

# Load all txt files
for file in os.listdir(data_folder):

    if file.endswith(".txt"):

        loader = TextLoader(
            os.path.join(data_folder, file),
            encoding="utf-8"
        )

        documents.extend(loader.load())

print(f"Loaded {len(documents)} documents")

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

print(f"Created {len(docs)} chunks")

# Create embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS vector database
vectorstore = FAISS.from_documents(
    docs,
    embeddings
)

# Save vector database locally
vectorstore.save_local("vectorstore/faiss_index")

print("FAISS vector database created successfully!")