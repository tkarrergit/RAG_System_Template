import os
from unstructured.partition.auto import partition
from vosk import Model, KaldiRecognizer
import wave
import numpy as np
import subprocess
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader, UnstructuredEPubLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from log_config import logger



class DocumentAudioProcessor:
    def __init__(self, model_path=f"assets/vosk-model-small-de-zamia-0.3"):
        logger.info("Initialisiere DocumentAudioProcessor")
        
        if not os.path.exists(model_path):
            logger.error("Vosk-Modell nicht gefunden: %s", model_path)
            raise ValueError("Vosk-Modell nicht gefunden. Laden Sie es herunter und geben Sie den Pfad an.")
        
        self.vosk_model = Model(model_path)
        logger.info("Vosk-Modell erfolgreich geladen")
        
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "cpu"})
        logger.info("Embedding-Modell erfolgreich geladen")
    
    def extract_text_from_pdf(self, file_path):
        logger.info("Extrahiere Text aus PDF: %s", file_path)
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        return "\n".join([page.page_content for page in pages])
    
    def extract_text_from_docx(self, file_path):
        logger.info("Extrahiere Text aus DOCX: %s", file_path)
        loader = UnstructuredWordDocumentLoader(file_path)
        document = loader.load()
        return document[0].page_content if document else ""
    
    def extract_text_from_epub(self, file_path):
        logger.info("Extrahiere Text aus EPUB: %s", file_path)
        loader = UnstructuredEPubLoader(file_path)
        document = loader.load()
        return "\n".join([doc.page_content for doc in document])
    
    def extract_text_from_txt(self, file_path):
        logger.info("Extrahiere Text aus TXT: %s", file_path)
        loader = TextLoader(file_path, encoding="utf-8")
        document = loader.load()
        return document[0].page_content if document else ""
    
    def extract_text_from_odt(self, file_path):
        logger.info("Extrahiere Text aus ODT: %s", file_path)
        elements = partition(filename=file_path)
        return "\n".join([str(el) for el in elements])
    
    def transcribe_audio(self, audio_path):
        logger.info("Transkribiere Audio: %s", audio_path)
        try:
            wf = wave.open(audio_path, "rb")
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
                logger.error("Ungültiges Audioformat: %s", audio_path)
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
            
            logger.info("Audio-Transkription abgeschlossen: %s", audio_path)
            return text
        except Exception as e:
            logger.exception("Fehler bei der Audio-Transkription: %s", str(e))
            raise
    
    def process_document(self, file_path):
        logger.info("Verarbeite Dokument: %s", file_path)
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
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
                logger.error("Nicht unterstütztes Dateiformat: %s", ext)
                raise ValueError("Nicht unterstütztes Dateiformat")
            
            logger.info("Dokumentenverarbeitung abgeschlossen: %s", file_path)
            return text
        except Exception as e:
            logger.exception("Fehler bei der Dokumentenverarbeitung: %s", str(e))
            raise
    
    def process_audio(self, audio_path):
        logger.info("Verarbeite Audio: %s", audio_path)
        try:
            text = self.transcribe_audio(audio_path)
            logger.info("Audioverarbeitung abgeschlossen: %s", audio_path)
            return text
        except Exception as e:
            logger.exception("Fehler bei der Audioverarbeitung: %s", str(e))
            raise