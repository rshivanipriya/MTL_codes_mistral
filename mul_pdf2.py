from langchain.document_loaders import UnstructuredPDFLoader
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
import os
import time

# Set Hugging Face token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# List of PDF files
pdf_files = ["8358-1997__jonewjudis20264.pdf", "8358-1997_jonewjudis20264.pdf", "8358-1997_jonewjudis_20264.pdf"]

# Initialize an empty list to store documents
all_docs = []

# Loop through PDF files
for pdf_file in pdf_files:
    # Load document
    loader = UnstructuredPDFLoader(pdf_file)
    document = loader.load()

    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, separators=[" ", ",", "\n"])
    docs = text_splitter.split_documents(document)

    # Extend the list of documents
    all_docs.extend(docs)


# Create FAISS index from embedded documents
embedding = HuggingFaceEmbeddings()
db = FAISS.from_documents(all_docs, embedding)

# Load Hugging Face LLM
llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", model_kwargs={"temperature": 0.2, "max_length": 1000})
chain = load_qa_chain(llm, chain_type="stuff")

# Perform similarity search based on query
query = "what is the case that happened on 28/10/1998"
start_time = time.time()  # Start measuring time
docs = db.similarity_search(query)
end_time = time.time()  # Stop measuring time
inference_time = end_time - start_time  # Calculate inference time for similarity search

# Run QA chain
start_time = time.time()  # Start measuring time
ans = chain.run(input_documents=docs, question=query)
end_time = time.time()  # Stop measuring time
qa_time = end_time - start_time  # Calculate inference time for QA chain

# Extract relevant text
full_text = ans
start_index = full_text.find("Helpful Answer:")
cropped_text = full_text[start_index + len("Helpful Answer:"):]

print(cropped_text)
print(f"Inference Time for Similarity Search: {inference_time} seconds")
print(f"Inference Time for QA Chain: {qa_time} seconds")
