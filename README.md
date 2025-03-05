# RAG System

## Status: Early Development / Beta Features

### Entwicklungszeitraum: Februar, 2025 bis März, 2025

---

## Technologien:

- **Python** – Hauptsprache des Projekts, ideal für schnelle Entwicklung und Flexibilität in der Dokumentenverarbeitung und -analyse.
- **Chromadb** – Datenbanktechnologie zur Speicherung von Embeddings und Text-Daten.
- **Hugging Face Embeddings** – Verwendung von vortrainierten Sprachmodellen zur Generierung von Dokumenten-Embeddings.
- **Vosk** – Offline-Spracherkennung für die Integration von Audio-Dokumenten.
- **Requests** – HTTP-Anfragen zum Kommunizieren mit dem LLM (Gemma 2 9B) für die Beantwortung von Fragen.
- **LangChain** – Framework zur Arbeit mit LLMs und Text-Dokumenten.
- **Flet** – Framework zur Entwicklung interaktiver GUIs (zukünftige Erweiterung möglich).

---

## Projektbeschreibung:

Das RAG System ist eine Anwendung, die es Nutzern ermöglicht, Dokumente zu verarbeiten und auf Grundlage dieser Dokumente Fragen zu stellen und Antworten zu generieren. Das System nutzt fortschrittliche Technologien wie **Spracherkennung** und **LLM-Integration**, um ein effizientes und interaktives Frage-Antwort-System zu bieten.

Dieses Projekt wurde entwickelt, um die automatische Verarbeitung von Textdokumenten mit modernen AI-Methoden zu demonstrieren und zu üben, insbesondere die Extraktion von Texten, die Generierung von Embeddings und die Nutzung eines **Retriever-Augmented Generation (RAG)** Systems zur Beantwortung von Fragen basierend auf gespeicherten Dokumenten.

---

## Lernfortschritte:

### Was ich gelernt habe:

- Erweiterte Kenntnisse in der Dokumentenverarbeitung mit Python, einschließlich der Extraktion von Text aus verschiedenen Formaten (PDF, DOCX, EPUB, etc.).
- Einführung in das Arbeiten mit Embedding-Modellen und deren Speicherung in Datenbanken (Chromadb).
- Integration eines Open-Source-Sprachmodells (Gemma 2 9B) zur Generierung von Antworten basierend auf Dokumenteninformationen.
- Nutzung von **LangChain** zur Verknüpfung von Dokumenten und LLMs für die Beantwortung von Fragen.
- Erfahrung in der Fehlerbehandlung und Optimierung der Performance bei der Verarbeitung großer Dokumentenmengen.

---

## Features:

### Aktuell Implementiert:

#### **Dokumentenverarbeitung**:
   - Text wird aus verschiedenen Dateiformaten (PDF, DOCX, EPUB, TXT, ODT) extrahiert.
   - Extrahierter Text wird in kleine Chunks aufgeteilt, um Embeddings zu erstellen.

#### **Embeddings**:
   - Erstellung und Speicherung von Embeddings basierend auf den Text-Chunks der Dokumente.
   - Vermeidung von Duplikaten durch Hashing der Chunks.

#### **Retriever-Augmented Generation (RAG)**:
   - Das System speichert Dokumenten-Embeddings und nutzt diese, um relevante Textabschnitte (Chunks) zu finden und als Kontext für die Antwortgenerierung zu verwenden.
   
#### **Frage-Antwort-System**:
   - Integration des LLM (Gemma 2 9B) zur Generierung von Antworten auf Anfragen, die auf den verarbeiteten Dokumenten basieren.
   - Das System nutzt den Kontext relevanter Chunks, um die Antwort präzise und relevant zu generieren.

---

## Hinweise zur Bibliotheksverwendung:

- **Chromadb**: Diese Datenbank wird für die Speicherung der Embeddings verwendet. Es ist darauf zu achten, dass regelmäßig Sicherungskopien der Datenbank erstellt werden, um Verlust von Dokumenten und Embeddings zu vermeiden.
  
- **Hugging Face Embeddings**: Die Einbindung von vortrainierten Modellen von Hugging Face wird benötigt, um Text-Embeddings zu erstellen. Bei neuen Modell-Versionen ist es sinnvoll, auf Updates zu achten.

- **LangChain**: Dieses Framework erleichtert die Verwaltung und Nutzung von Dokumenten und Modellen im Zusammenhang mit LLMs. Falls neue Funktionen von LangChain verfügbar sind, sollten diese geprüft und bei Bedarf integriert werden.

- **Vosk (Spracherkennung)**: Diese Bibliothek dient der Offline-Spracherkennung und wird in Zukunft für die Verarbeitung von Audio-Dokumenten genutzt.

---

## Hinweise zu den bestehenden Features:

#### **Textverarbeitung**:
   - Der `process_document()`-Prozess extrahiert Text aus verschiedenen Formaten (PDF, DOCX, EPUB, TXT, ODT) und sorgt für eine einheitliche Darstellung des Inhalts.
   - Die `text_splitter()`-Methode sorgt dafür, dass der extrahierte Text in handhabbare Teile (Chunks) zerlegt wird, die dann mit Embeddings versehen werden.

#### **Datenbank-Management**:
   - **Chromadb** speichert die Embeddings und stellt sicher, dass keine doppelten Text-Chunks hinzugefügt werden, indem jeder Chunk gehasht wird.
   - Die `retrieve_relevant_chunks()`-Funktion sucht nach den relevantesten Textteilen, basierend auf einer gegebenen Anfrage.

#### **LLM-Integration**:
   - Das System nutzt die **Gemma 2 9B**-API, um Antworten zu generieren, basierend auf dem Kontext der abgerufenen Text-Chunks.

---

## Zukünftige Verbesserungen:

- **Optimierung der Performance** bei der Verarbeitung von großen Dokumenten und beim Abrufen von relevanten Chunks.
- **Erweiterung der Datenbank** zur Unterstützung zusätzlicher Dokumententypen und -formatierungen.
- **Benutzeroberfläche (UI)**: Implementierung einer UI.

