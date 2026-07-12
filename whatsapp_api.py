from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn, json, logging
from whatsapp_handler import main as handle_msg
import sys

logging.basicConfig(level=logging.INFO)
app = FastAPI()

@app.post("/whatsapp-drive")
async def receive_whatsapp(request: Request):
    try:
        body = await request.json()
        value = body["entry"][0]["changes"][0]["value"]
        messages = value.get("messages", [])
        if not messages:
            return JSONResponse({"status": "no_message"})
        msg = messages[0]
        phone = msg["from"]
        msg_type = msg["type"]
        contacts = value.get("contacts", [])
        name = contacts[0]["profile"]["name"] if contacts else "לקוח"
        
        from whatsapp_drive import save_message_to_drive
        if msg_type == "text":
            save_message_to_drive(phone, name, "text", msg["text"]["body"])
        else:
            filename = msg.get(msg_type, {}).get("filename", f"{msg_type}.bin")
            save_message_to_drive(phone, name, msg_type, b"", filename)
        
        return JSONResponse({"status": "ok", "client": name, "type": msg_type})
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
