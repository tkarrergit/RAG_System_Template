import rag_system_tools as rst

rag = rst.RAGSystemTools()

def process_files(file_paths):
    for file_path in file_paths:
        text = rag.process_document(file_path)
        chunks = rag.text_splitter(text)
        embeddings = rag.create_embedding(chunks)
        rag.store_embeddings(chunks, embeddings)

file_paths = rag.process_documents_paths()
process_files(file_paths)

# ✅ Frage stellen und Antwort von Gemma 2 7B erhalten
frage = "Worum geht es in dem Dokument?"
antwort = rag.generate_answer(frage)
print("\n💡 KI-Antwort:", antwort)