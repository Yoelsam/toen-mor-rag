import sys, json, logging
from whatsapp_drive import save_message_to_drive, get_or_create_folder

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def main():
    if len(sys.argv) < 4:
        print("Usage: whatsapp_handler.py <phone> <type> <json_body>")
        sys.exit(1)
    
    phone = sys.argv[1].strip("'")
    msg_type = sys.argv[2].strip("'")
    body = json.loads(sys.argv[3].strip("'"))
    
    try:
        value = body["entry"][0]["changes"][0]["value"]
        messages = value.get("messages", [])
        contacts = value.get("contacts", [])
        if not messages:
            print("Pas de message")
            return
        msg = messages[0]
        name = contacts[0]["profile"]["name"] if contacts else "לקוח"
        log.info(f"Message de {name} ({phone}), type: {msg_type}")
        if msg_type == "text":
            text = msg["text"]["body"]
            save_message_to_drive(phone, name, "text", text)
            print(f"OK - texte sauvegardé pour {name}")
        elif msg_type in ["document", "image", "audio", "video"]:
            filename = msg.get(msg_type, {}).get("filename", f"{msg_type}.bin")
            save_message_to_drive(phone, name, msg_type, b"", filename)
            print(f"OK - fichier {filename} sauvegardé pour {name}")
        else:
            print(f"Type non géré: {msg_type}")
    except Exception as e:
        log.error(f"Erreur: {e}")
        print(f"ERREUR: {e}")

if __name__ == "__main__":
    main()
