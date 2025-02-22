import requests
import chromadb
import numpy as np
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 🔹 Logging einrichten
logging.basicConfig(
    filename="rag_system.log",  # Log-Datei
    level=logging.DEBUG,  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)  # In Konsole auch DEBUG ausgeben
formatter = logging.Formatter("[%(levelname)s] %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)  # Log-Ausgabe in Konsole aktivieren

# 🔹 LLM-Anbindung über LM Studio API
class LLMClient:
    def __init__(self, api_url="http://localhost:1234/v1/chat/completions"):
        self.api_url = api_url

    def generate(self, context, query):
        prompt = f"Kontext: {context}\n\nFrage: {query}\nAntwort:"
        logging.debug(f"📄 Kontext an das LLM: {context[:200]}...")  # Nur ersten 200 Zeichen loggen
        response = requests.post(
            self.api_url,
            json={
                "model": "gemma-2-9b-it",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 200
            }
        )
        if response.status_code == 200:
            logging.info("✅ LLM hat erfolgreich geantwortet.")
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", "Fehler: Keine Antwort erhalten.")
        else:
            logging.error(f"❌ Fehler beim LLM-Request: {response.status_code}")
            return "Fehler: Keine Antwort erhalten."

class RAGSystem:
    def __init__(self, db_path="./vector_db"):
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "cpu"})
        self.db = chromadb.PersistentClient(path=db_path)
        self.collection = self.db.get_or_create_collection("documents")
        self.llm = LLMClient()
        logging.info("✅ RAGSystem initialisiert.")

    def extract_text_from_pdf(self, pdf_path):
        logging.info(f"📂 Extrahiere Text aus PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        text = "\n".join([page.page_content for page in pages])
        logging.debug(f"📄 Extrahierter Text (erster 200 Zeichen): {text[:200]}...")
        return text

    def create_embedding(self, text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        texts = text_splitter.split_text(text)
        embeddings = self.embedding_model.embed_documents(texts)
        logging.info(f"✅ {len(texts)} Text-Chunks erstellt und eingebettet.")
        return texts, np.array(embeddings)

    def store_embeddings(self, texts, embeddings):
        existing_ids = len(self.collection.get()["ids"]) if self.collection.get() else 0
        for i, (chunk, embedding) in enumerate(zip(texts, embeddings)):
            self.collection.add(
                ids=[str(i + existing_ids)],  # IDs fortlaufend vergeben
                documents=[chunk],
                embeddings=[embedding.tolist()]
            )
        logging.info(f"✅ {len(texts)} Chunks in die Vektordatenbank gespeichert.")
        logging.debug(f"📌 Anzahl gespeicherter Embeddings: {len(self.collection.get()['documents'])}")

    def retrieve_relevant_chunks(self, query, n_results=3):
        query_embedding = self.embedding_model.embed_query(query)
        max_results = min(n_results, len(self.collection.get()["documents"]))  # Falls zu wenig Elemente in DB, begrenzen
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max_results
        )
        logging.info(f"🔍 {len(results['documents'])} relevante Chunks gefunden.")
        logging.debug(f"📄 Gefundene Chunks: {results['documents']}")
        return results["documents"][0] if results["documents"] else []

    def generate_answer(self, query):
        relevant_chunks = self.retrieve_relevant_chunks(query)
        context = " ".join(relevant_chunks)
        return self.llm.generate(context, query)
