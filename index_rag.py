import os, io, pickle, time, logging
from pathlib import Path
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import google.generativeai as genai
import chromadb
from docx import Document
import PyPDF2

# CONFIG
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GDRIVE_FOLDER = "PsakeiDin_Rabbaniyim"
SCOPES = ["https://www.googleapis.com/auth/drive"]
CHROMA_DIR = "/home/rag/chroma_db"
COLLECTION_NAME = "psakei_din"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

genai.configure(api_key=GEMINI_API_KEY)

def get_gdrive_service():
    with open("/home/rag/token.pickle", "rb") as f:
        creds = pickle.load(f)
    if not creds.valid and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open("/home/rag/token.pickle", "wb") as f:
            pickle.dump(creds, f)
    return build("drive", "v3", credentials=creds)

def get_folder_id(service, name):
    q = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    files = service.files().list(q=q, fields="files(id)").execute().get("files", [])
    return files[0]["id"] if files else None

def list_files(service, folder_id):
    files = []
    page_token = None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token, pageSize=100
        ).execute()
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files

def download_file(service, file_id):
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return fh

def extract_text_docx(fh):
    try:
        doc = Document(fh)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        log.error(f"Erreur docx: {e}")
        return ""

def extract_text_pdf(fh):
    try:
        reader = PyPDF2.PdfReader(fh)
        return "\n".join(page.extract_text() for page in reader.pages)
    except Exception as e:
        log.error(f"Erreur pdf: {e}")
        return ""

def chunk_text(text, size=2000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def embed_text(text):
    try:
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text[:8000],
            task_type="retrieval_document"
        )
        return result["embedding"]
    except Exception as e:
        log.error(f"Erreur embedding: {e}")
        return None

def main():
    if not GEMINI_API_KEY:
        log.error("GEMINI_API_KEY manquante! Lancez: export GEMINI_API_KEY='votre_cle'")
        return

    log.info("Connexion Google Drive...")
    service = get_gdrive_service()
    folder_id = get_folder_id(service, GDRIVE_FOLDER)
    if not folder_id:
        log.error("Dossier introuvable!")
        return

    log.info("Liste des fichiers...")
    files = list_files(service, folder_id)
    log.info(f"{len(files)} fichiers trouves")

    # ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    total_chunks = 0
    for i, f in enumerate(files):
        name = f["name"]
        ext = Path(name).suffix.lower()
        try:
            fh = download_file(service, f["id"])
            if ext == ".docx":
                text = extract_text_docx(fh)
            elif ext == ".pdf":
                text = extract_text_pdf(fh)
            else:
                continue

            if not text.strip():
                continue

            chunks = chunk_text(text)
            for j, chunk in enumerate(chunks):
                emb = embed_text(chunk)
                if emb:
                    collection.add(
                        ids=[f"{f['id']}_{j}"],
                        embeddings=[emb],
                        documents=[chunk],
                        metadatas=[{"filename": name, "chunk": j}]
                    )
                    total_chunks += 1
                time.sleep(0.05)

            if (i + 1) % 10 == 0:
                log.info(f"[{i+1}/{len(files)}] {total_chunks} chunks indexes")
        except Exception as e:
            log.error(f"Erreur fichier {name}: {e}")

    log.info(f"=== Termine! {total_chunks} chunks indexes dans ChromaDB ===")

if __name__ == "__main__":
    main()
