import json, os, io, pickle, logging
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from google.auth.transport.requests import Request

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

CLIENTS_DB = "/home/rag/whatsapp_clients.json"
TOKEN_PATH = "/home/rag/token.pickle"

def get_drive_service():
    with open(TOKEN_PATH, "rb") as f:
        creds = pickle.load(f)
    if not creds.valid and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return build("drive", "v3", credentials=creds)

def load_clients():
    with open(CLIENTS_DB) as f:
        return json.load(f)

def save_clients(data):
    with open(CLIENTS_DB, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def is_group(phone):
    return phone.endswith("@g.us")

def get_or_create_folder(phone, name=None):
    service = get_drive_service()
    data = load_clients()
    group = is_group(phone)
    parent_id = data["groups_folder_id"] if group else data["clients_folder_id"]
    prefix = "👥" if group else "👤"
    
    if phone in data["clients"]:
        return data["clients"][phone]
    
    folder_name = f"{prefix} {name or 'חדש'} - {phone.replace('@g.us','').replace('@s.whatsapp.net','')}"
    folder = service.files().create(
        body={"name": folder_name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]},
        fields="id,name"
    ).execute()
    
    data["clients"][phone] = {
        "name": name or "חדש",
        "folder_id": folder["id"],
        "type": "group" if group else "client",
        "added": datetime.now().isoformat()
    }
    save_clients(data)
    log.info(f"Nouveau {'groupe' if group else 'client'}: {folder_name}")
    return data["clients"][phone]

def save_message_to_drive(phone, name, msg_type, content, filename=None):
    service = get_drive_service()
    entry = get_or_create_folder(phone, name)
    folder_id = entry["folder_id"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    if msg_type == "text":
        fname = f"{now} - הודעה.txt"
        media = MediaIoBaseUpload(io.BytesIO(content.encode("utf-8")), mimetype="text/plain")
    else:
        fname = f"{now} - {filename or msg_type}"
        media = MediaIoBaseUpload(io.BytesIO(content if isinstance(content, bytes) else b""), mimetype="application/octet-stream")
    
    service.files().create(
        body={"name": fname, "parents": [folder_id]},
        media_body=media
    ).execute()
    log.info(f"Sauvegardé: {fname}")
    return fname

if __name__ == "__main__":
    result = get_or_create_folder("0559155960", "אהוד נתן מור")
    print("Test OK:", result)
