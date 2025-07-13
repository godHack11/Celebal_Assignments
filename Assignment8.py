# RAG Q&A chatbot
# RAG Q&A chatbot using document retrieval and generative AI for intelligent response generation (can use any light model from hugging face or a license llm(opneai, claude, grok, gemini)
# if free credits available


import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = "sk-***********************************"  

client = OpenAI()

def load_loan_data(path='Training Dataset.csv'):
    df = pd.read_csv(path)
    df = df.astype(str)
    df.fillna("Unknown", inplace=True)
    return df
def df_to_documents(df):
    return ["\n".join([f"{col}: {val}" for col, val in row.items()]) for _, row in df.iterrows()]

def store_documents_in_chroma(documents):
    chroma_client = chromadb.Client()
    try:
        chroma_client.delete_collection("loan_data")
    except:
        pass
    collection = chroma_client.create_collection(name="loan_data")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(documents)
    for i, (doc, emb) in enumerate(zip(documents, embeddings)):
        collection.add(documents=[doc], embeddings=[emb.tolist()], ids=[str(i)])
    return collection
def openai_generate(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for answering loan-related queries."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )
    return response.choices[0].message.content

def answer_with_context(query, collection):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    context = "\n".join(results['documents'][0])
    prompt = f"""Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}\nAnswer:"""
    return openai_generate(prompt)

if __name__ == "__main__":
    print("Script is running...")
    df = load_loan_data()
    print(f"Dataset loaded: {len(df)} rows")
    docs = df_to_documents(df)
    print("Converted rows into text documents")
    print("Storing documents in ChromaDB...")
    collection = store_documents_in_chroma(docs)
    print("Embeddings stored in vector database\n")
    print("Chatbot is ready! Type your questions below.\n")

    while True:
        query = input("Ask any question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        try:
            print("You asked:", query)
            answer = answer_with_context(query, collection)
            print(" ", answer.strip())
        except Exception as e:
            print("Error while generating answer:", e)
