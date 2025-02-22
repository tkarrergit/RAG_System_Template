import os
from unstructured.partition.auto import partition
from vosk import Model, KaldiRecognizer
import wave
import numpy as np
import subprocess
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader, UnstructuredEPubLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter



class DocumentAudioProcessor:
    def __init__(self, model_path=f"assets/vosk-model-small-de-zamia-0.3"):
        # Initialisiere das Vosk-Spracherkennungsmodell
        if not os.path.exists(model_path):
            raise ValueError("Vosk-Modell nicht gefunden. Laden Sie es herunter und geben Sie den Pfad an.")
        
        self.vosk_model = Model(model_path)
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "cpu"})
    
    def extract_text_from_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        return "\n".join([page.page_content for page in pages])
    
    def extract_text_from_docx(self, file_path):
        loader = UnstructuredWordDocumentLoader(file_path)
        document = loader.load()
        return document[0].page_content if document else ""
    
    def extract_text_from_epub(self, file_path):
        loader = UnstructuredEPubLoader(file_path)
        document = loader.load()
        return "\n".join([doc.page_content for doc in document])
    
    def extract_text_from_txt(self, file_path):
        loader = TextLoader(file_path, encoding="utf-8")
        document = loader.load()
        return document[0].page_content if document else ""
    
    def extract_text_from_odt(self, file_path):
        elements = partition(filename=file_path)
        return "\n".join([str(el) for el in elements])
    
    def transcribe_audio(self, audio_path):
        wf = wave.open(audio_path, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            raise ValueError("Audio muss ein WAV-Format mit 16-Bit PCM Mono sein.")
        
        recognizer = KaldiRecognizer(self.vosk_model, wf.getframerate())
        recognizer.SetWords(True)
        
        text = ""
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                text += recognizer.Result()
        
        return text
    
    def create_embedding(self, text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        texts = text_splitter.split_text(text)
        embeddings = self.embedding_model.embed_documents(texts)
        return np.array(embeddings)
    
    def process_document(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            text = self.extract_text_from_pdf(file_path)
        elif ext == ".docx":
            text = self.extract_text_from_docx(file_path)
        elif ext == ".epub":
            text = self.extract_text_from_epub(file_path)
        elif ext == ".txt":
            text = self.extract_text_from_txt(file_path)
        elif ext == ".odt":
            text = self.extract_text_from_odt(file_path)
        else:
            raise ValueError("Nicht unterstütztes Dateiformat")
        
        return text, self.create_embedding(text)
    
    def create_embedding(self, text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        texts = text_splitter.split_text(text)
        embeddings = self.embedding_model.embed_documents(texts)
        return np.array(embeddings)
    
    def process_document(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            text = self.extract_text_from_pdf(file_path)
        elif ext == ".docx":
            text = self.extract_text_from_docx(file_path)
        elif ext == ".epub":
            text = self.extract_text_from_epub(file_path)
        elif ext == ".txt":
            text = self.extract_text_from_txt(file_path)
        elif ext == ".odt":
            text = self.extract_text_from_odt(file_path)
        else:
            raise ValueError("Nicht unterstütztes Dateiformat")
        
        return text, self.create_embedding(text)
    
    def process_audio(self, audio_path):
        text = self.transcribe_audio(audio_path)
        return text, self.create_embedding(text)