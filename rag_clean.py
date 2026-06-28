import os, io, pickle, logging
from pathlib import Path
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import chromadb
from chromadb.utils import embedding_functions
from docx import Document
import PyPDF2

GDRIVE_FOLDER = "PsakeiDin_Rabbaniyim"
CHROMA_DIR = "/home/rag/chroma_db"
COLLECTION_NAME = "psakei_din"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def get_gdrive_service():
    with open("/home/rag/token.pickle", "rb") as f:
        creds = pickle.load(f)
    if not creds.valid and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return build("drive", "v3", credentials=creds)

def get_folder_id(service, name):
    q = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    files = service.files().list(q=q, fields="files(id)").execute().get("files", [])
    return files[0]["id"] if files else None

def list_files(service, folder_id):
    files = []
    token = None
    while True:
        resp = service.files().list(q=f"'{folder_id}' in parents and trashed=false", fields="nextPageToken, files(id, name)", pageToken=token, pageSize=100).execute()
        files.extend(resp.get("files", []))
        token = resp.get("nextPageToken")
        if not token:
            break
    return files

def download_file(service, file_id):
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    dl = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = dl.next_chunk()
    fh.seek(0)
    return fh

def extract_docx(fh):
    try:
        doc = Document(fh)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception:
        return ""

def extract_pdf(fh):
    try:
        reader = PyPDF2.PdfReader(fh)
        return "\n".join(page.extract_text() for page in reader.pages)
    except Exception:
        return ""

def chunk_text(text, size=1500, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+size])
        start += size - overlap
    return chunks

def main():
    log.info("Connexion Google Drive...")
    service = get_gdrive_service()
    folder_id = get_folder_id(service, GDRIVE_FOLDER)
    if not folder_id:
        log.error("Dossier introuvable!")
        return
    files = list_files(service, folder_id)
    log.info(f"{len(files)} fichiers trouves")
    log.info("Chargement du modele embedding local...")
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass
    collection = client.create_collection(COLLECTION_NAME, embedding_function=embed_fn)
    total = 0
    for i, f in enumerate(files):
        name = f["name"]
        ext = Path(name).suffix.lower()
        try:
            fh = download_file(service, f["id"])
            if ext == ".docx":
                text = extract_docx(fh)
            elif ext == ".pdf":
                text = extract_pdf(fh)
            else:
                continue
            if not text.strip():
                continue
            chunks = chunk_text(text)
            ids = [f"{f['id']}_{j}" for j in range(len(chunks))]
            metas = [{"filename": name} for _ in chunks]
            collection.add(ids=ids, documents=chunks, metadatas=metas)
            total += len(chunks)
            if (i+1) % 25 == 0:
                log.info(f"[{i+1}/{len(files)}] {total} chunks")
        except Exception as e:
            log.error(f"Erreur {name}: {e}")
    log.info(f"=== Termine! {total} chunks indexes ===")

if __name__ == "__main__":
    main()
