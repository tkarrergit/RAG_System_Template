import requests
import chromadb
from document_audio_processor import DocumentAudioProcessor
from log_config import logger
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import os
import hashlib


# 🔹 LLM-Anbindung über LM Studio API
class LLMClient:
    def __init__(self, api_url="http://localhost:1234/v1/chat/completions"):
        self.api_url = api_url

    def generate(self, context, query):
        prompt = f"Kontext: {context}\n\nFrage: {query}\nAntwort:"
        logger.debug(f"📄 Kontext an das LLM: {context[:200]}...")  # Nur ersten 200 Zeichen loggen
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
            logger.info("✅ LLM hat erfolgreich geantwortet.")
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", "Fehler: Keine Antwort erhalten.")
        else:
            logger.error(f"❌ Fehler beim LLM-Request: {response.status_code}")
            return "Fehler: Keine Antwort erhalten."

class RAGSystemTools(DocumentAudioProcessor):
    def __init__(self, db_path="./vector_db"):
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "cpu"})
        self.db = chromadb.PersistentClient(path=db_path)
        self.collection = self.db.get_or_create_collection("documents")
        self.llm = LLMClient()
        self.existing_hashes = set(self.load_existing_hashes())  # ✅ Lade bestehende Hashes
        logger.info("✅ RAGSystem erfolgreich initialisiert.")

    def hash_text(self, text):
        """Erstellt einen SHA256-Hash für den Text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def load_existing_hashes(self):
        """Lädt alle gespeicherten Dokumenten-Hashes."""
        hashes = set()
        existing_docs = self.collection.get()
        if existing_docs and "documents" in existing_docs:
            for doc in existing_docs["documents"]:
                hashes.add(self.hash_text(doc))
        logger.info(f"📌 {len(hashes)} bestehende Dokumente erkannt.")
        return hashes

    def text_splitter(self, text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        chunks = text_splitter.split_text(text)
        logger.info(f"✅ {len(chunks)} Text-Chunks erstellt.")
        return chunks

    def create_embedding(self, chunks):
        embeddings = self.embedding_model.embed_documents(chunks)
        logger.info(f"✅ {len(chunks)} Text-Chunks eingebettet.")
        return np.array(embeddings)
   
    def store_embeddings(self, texts, embeddings):
        """Speichert neue Embeddings in der Datenbank, vermeidet doppelte Inhalte."""
        new_chunks = []
        new_embeddings = []

        for chunk, embedding in zip(texts, embeddings):
            chunk_hash = self.hash_text(chunk)

            if chunk_hash not in self.existing_hashes:
                new_chunks.append(chunk)
                new_embeddings.append(embedding)
                self.existing_hashes.add(chunk_hash)  # ✅ Neuen Hash speichern
            else:
                logger.warning(f"⚠️ Doppelte Einfügung erkannt und übersprungen: {chunk[:100]}...")  # Nur 100 Zeichen loggen

        if new_chunks:
            existing_ids = len(self.collection.get()["ids"]) if self.collection.get() else 0
            for i, (chunk, embedding) in enumerate(zip(new_chunks, new_embeddings)):
                self.collection.add(
                    ids=[str(i + existing_ids)], 
                    documents=[chunk],
                    embeddings=[embedding.tolist()]
                )
            logger.info(f"✅ {len(new_chunks)} neue Chunks gespeichert.")
        else:
            logger.info("🚀 Keine neuen Chunks zu speichern – alles bereits vorhanden.")
       
    def retrieve_relevant_chunks(self, query, n_results=3):
        query_embedding = self.embedding_model.embed_query(query)
        max_results = min(n_results, len(self.collection.get()["documents"]))  # Falls zu wenig Elemente in DB, begrenzen
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max_results
        )
        logger.info(f"🔍 {len(results['documents'])} relevante Chunks gefunden.")
        logger.debug(f"📄 Gefundene Chunks: {results['documents']}")
        return results["documents"][0] if results["documents"] else []

    def generate_answer(self, query):
        relevant_chunks = self.retrieve_relevant_chunks(query)
        context = " ".join(relevant_chunks)
        return self.llm.generate(context, query)
