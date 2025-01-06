# needs langchain-community pypdf
# pip install langchain-community pypdf
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

if __name__ == '__main__':
    load_dotenv()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    file_path = "sec-hot-topics.pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(len(docs))
    print(f"{docs[0].page_content[:200]}\n")
    print(docs[0].metadata)

    # why do we need text splitter? can we just feed the whole docs into model?
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    print(len(all_splits))

    vector_1 = embeddings.embed_query(all_splits[0].page_content)
    vector_2 = embeddings.embed_query(all_splits[1].page_content)

    assert len(vector_1) == len(vector_2)
    print(f"Generated vectors of length {len(vector_1)}\n")
    print(vector_1[:10])

    vector_store = InMemoryVectorStore(embeddings)
    ids = vector_store.add_documents(documents=all_splits)

    results = vector_store.similarity_search(
        "What is the balance due?"
    )
    print("------ Query: What is the balance due?")
    print(results[0])
    print()

    embedding = embeddings.embed_query("What is Meijun Yin's phone number?")
    results = vector_store.similarity_search_by_vector(embedding)
    print("------ Query: What is Meijun Yin's phone number?")
    print(results[0])
    print()

    @chain
    def retriever(query: str) -> list[Document]:
        return vector_store.similarity_search(query, k=1)

    results = retriever.batch(
        [
            "What is the balance due?",
            "What is Meijun Yin's phone number?"
        ]
    )
    print("------ Documents from retriever")
    print(results)
    print()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )
    results = retriever.batch(
        [
            "What is the balance due?",
            "What is Meijun Yin's phone number?"
        ]
    )
    print("------ Documents from retriever")
    print(results)
    print()
