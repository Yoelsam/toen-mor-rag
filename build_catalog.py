#!/usr/bin/env python3
import chromadb, json, time, anthropic, os

client_ai = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", open("/home/rag/rag.env").read().split("ANTHROPIC_API_KEY=")[1].split()[0].strip()))
c = chromadb.PersistentClient(path="/home/rag/chroma_db")
col = c.get_collection("psakei_din")

# Get all unique files
r = col.get(limit=10000, include=["metadatas","documents"])
seen = {}
for doc, meta in zip(r["documents"], r["metadatas"]):
    fn = meta.get("filename","")
    if fn not in seen:
        seen[fn] = {"filename": fn, "file_id": meta.get("file_id",""), "text": doc[:1500]}

print(f"Found {len(seen)} unique files")

catalog = []
done = 0
errors = 0

for fn, info in seen.items():
    try:
        prompt = f"""נתח את המסמך המשפטי הבא והחזר JSON בלבד (בלי טקסט נוסף):
{{"tik": "מספר תיק אם יש, אחרת null",
"erakha": "ערכאה (בית משפט עליון/מחוזי/משפחה/בית דין גדול/אזורי/אחר) אם יש, אחרת null",
"date": "תאריך בפורמט DD.MM.YYYY אם יש, אחרת null",
"type": "סוג (פסק_דין/מאמר/חקיקה/שות/ספר/אחר)",
"topic": "נושא בקצרה (עד 10 מילים)",
"judges": "שמות שופטים/דיינים אם יש, אחרת null"}}

מסמך:
{info['text'][:1000]}"""

        msg = client_ai.messages.create(model="claude-haiku-4-5-20251001", max_tokens=300, messages=[{"role":"user","content":prompt}])
        txt = next((b.text for b in msg.content if hasattr(b,'text')), "{}")
        # Extract JSON
        start = txt.find("{")
        end = txt.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(txt[start:end])
            data["filename"] = fn
            data["file_id"] = info["file_id"]
            catalog.append(data)
        done += 1
        if done % 50 == 0:
            print(f"Done {done}/{len(seen)}")
            json.dump(catalog, open("/home/rag/index_juridique.json","w"), ensure_ascii=False, indent=2)
        time.sleep(0.5)
    except Exception as e:
        errors += 1
        if errors < 5:
            print(f"Error on {fn[:30]}: {e}")
        done += 1

json.dump(catalog, open("/home/rag/index_juridique.json","w"), ensure_ascii=False, indent=2)
print(f"DONE: {len(catalog)} files catalogued, {errors} errors")
