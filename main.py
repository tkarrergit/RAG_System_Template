import document_audio_processor as dap
import rag_system_tools as rs
import os
"""
processor = dap.DocumentAudioProcessor()

# Beispiel: Verarbeitung eines PDF-Dokuments
base_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(base_dir, "documents/example.pdf")
text, embedding = processor.process_document(pdf_path)
print("Extrahierter Text aus PDF:", text[:500])  # Zeigt einen Ausschnitt des extrahierten Texts
print("Embedding-Shape:", embedding.shape)

# Beispiel: Verarbeitung einer Audioaufnahme
audio_path = "example.wav"
transcribed_text, audio_embedding = processor.process_audio(audio_path)
print("Transkribierter Text:", transcribed_text)
print("Audio-Embedding-Shape:", audio_embedding.shape)"""

# ✅ Beispiel: PDF verarbeiten
rag = rs.RAGSystemTools()
dap = dap.DocumentAudioProcessor()
base_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(base_dir, "documents/")
extensions = [".pdf", ".docx", ".epub", ".txt", ".odt"]
file_paths = dap.create_path_of_files_with_extensions(folder_path, extensions)
print(file_paths)

#pdf_path = os.path.join(base_dir, "documents/example.pdf")
text = dap.process_document(pdf_path)
chunks = rag.text_splitter(text)
embeddings = rag.create_embedding(chunks)
rag.store_embeddings(chunks, embeddings)

# ✅ Frage stellen und Antwort von Gemma 2 7B erhalten
frage = "Worum geht es in dem Dokument?"
antwort = rag.generate_answer(frage)
print("\n💡 KI-Antwort:", antwort)