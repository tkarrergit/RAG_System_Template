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
from odf.opendocument import load
from odf.text import P
import shutil


class DocumentAudioProcessor:
    def __init__(self, model_path=f"assets/vosk-model-small-de-zamia-0.3"):
        logger.info("Initialisiere DocumentAudioProcessor")
        
        if not os.path.exists(model_path):
            logger.error("Vosk-Modell nicht gefunden: %s", model_path)
            raise ValueError("Vosk-Modell nicht gefunden. Laden Sie es herunter und geben Sie den Pfad an.")
        
        self.vosk_model = Model(model_path)
        logger.info("Vosk-Modell erfolgreich geladen")
    
    def create_path_of_files_with_extensions(self, folder_path, extensions):
        """
        Sucht in einem Ordner nach Dateien mit bestimmten Endungen und gibt eine Liste der Dateipfade zurück.

        :param folder_path: Pfad zum Ordner, in dem gesucht werden soll.
        :param extensions: Liste der Dateiendungen, nach denen gesucht werden soll (z.B. ['.pdf', '.docx']).
        :return: Liste der Dateipfade.
        """
        logger.info("Suche nach Dateien in Ordner: %s mit Endungen: %s", folder_path, extensions)
        file_paths = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    file_paths.append(os.path.join(root, file))
                    logger.debug("Gefundene Datei: %s", os.path.join(root, file))
        
        logger.info("Suche abgeschlossen. Gefundene Dateien: %d", len(file_paths))
        logger.info(f"File Pfade sind: {file_paths}")
        return file_paths

    def move_processed_file(self, file_path: str, target_dir: str) -> str:
        """
        Verschiebt eine Datei in ein bestimmtes Verzeichnis.
        Erstellt das Zielverzeichnis, falls es nicht existiert.

        :param file_path: Der vollständige Pfad der Datei, die verschoben werden soll.
        :param target_dir: Das Zielverzeichnis, in das die Datei verschoben wird.
        :return: Der neue Dateipfad.
        :raises FileNotFoundError: Falls die Datei nicht existiert.
        """

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Die Datei '{file_path}' existiert nicht.")

        os.makedirs(target_dir, exist_ok=True)  # Erstelle das Verzeichnis, falls es nicht existiert

        new_path = os.path.join(target_dir, os.path.basename(file_path))
        shutil.move(file_path, new_path)
        logger.info(f"Datei verschoben: {file_path} -> {new_path}")

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
        """
        Extrahiert den Text aus einer ODT-Datei.
        """
        try:
            logger.info("Extrahiere Text aus ODT: %s", file_path)
            odt_doc = load(file_path)
            text_content = []
            
            for element in odt_doc.getElementsByType(P):
                text_content.append(element.firstChild.data if element.firstChild else "")
            
            return "\n".join(text_content)
        except Exception as e:
            logger.error("Fehler beim Extrahieren des Textes: %s", str(e))
            return ""
    
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