import re
import time
import os, pickle, logging
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from google import genai
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import io
from docx import Document

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
CHROMA_DIR = "/home/rag/chroma_db"
COLLECTION_NAME = "psakei_din"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

client_gemini = genai.Client(api_key=GEMINI_API_KEY)

embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(COLLECTION_NAME)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
import auth, database
from starlette.middleware.sessions import SessionMiddleware
app.add_middleware(SessionMiddleware,secret_key=os.environ["SESSION_SECRET"],https_only=True,same_site="lax",max_age=14*24*3600)
app.include_router(auth.router)
database.init_db()

def search_docs(query, n=10):
    results = collection.query(query_texts=[query], n_results=60)
    docs_all = results["documents"][0]
    metas_all = results["metadatas"][0]
    seen = set()
    docx_list = []
    pdf_list = []
    for d, m in zip(docs_all, metas_all):
        fn = m.get("filename", "")
        if fn not in seen:
            seen.add(fn)
            if fn.endswith(".docx"):
                docx_list.append((d, m))
            elif fn.endswith(".pdf"):
                pdf_list.append((d, m))
    mixed = docx_list[:7] + pdf_list[:3]
    docs_dedup = [x[0] for x in mixed]
    metas_dedup = [x[1] for x in mixed]
    return docs_dedup, metas_dedup

def ask_gemini(query, docs, metas, history=None):
    context = "\n\n".join(f"[\u05DE\u05E7\u05D5\u05E8 {i+1}: {m['filename']}]\n{d}" for i,(d,m) in enumerate(zip(docs,metas)))
    history_text = ""
    if history:
        for h in history[-4:]:
            history_text += f"\n\u05E9\u05D0\u05DC\u05D4 \u05E7\u05D5\u05D3\u05DE\u05EA: {h['q']}\n\u05EA\u05E9\u05D5\u05D1\u05D4 \u05E7\u05D5\u05D3\u05DE\u05EA: {h['a'][:500]}\n---"
    prompt = f"""אתה עוזר משפטי מומחה המתמחה בדיני משפחה, דיני עבודה ודיני רבניים בישראל.
תפקידך לענות על שאלות משפטיות בצורה מפורטת, מדויקת ומועילה.

כללי תשובה:
1. השתמש במקורות שסופקו כבסיס עיקרי לתשובתך.
2. אם המקורות מכילים מידע רלוונטי חלקי - השתמש בו וציין זאת.
3. אם המקורות אינם מכילים תשובה ישירה - ענה על בסיס עקרונות הלכתיים ומשפטיים כלליים הידועים לך, וציין במפורש שהתשובה מבוססת על ידע כללי ולא על פסיקה ספציפית מהמאגר.
4. לעולם אל תאמר "אין תשובה" - תמיד ספק את המידע הטוב ביותר שיש לך.
5. CRITICAL: בכל טענה ציין את מספר המקור בפורמט המדויק [מקור X]
0. אסור בהחלט ליצור קישורים לאינטרנט, לגוגל דרייב, או כל URL אחר בתשובה. - X הוא המספר שמופיע לפני שם הקובץ במקורות למטה.
6. דוגמה לתשובה נכונה: "הבעל חייב במזונות אשתו [מקור 1]. עם זאת, אם האישה מורדת, הוא פטור [מקור 3, מקור 4]."
7. חובה להשתמש במקורות שונים - לא רק מקור אחד. בדוק כל מקור וציין אותו בנפרד.
8. מבנה התשובה: עיקר הדין / ניתוח המקרה / המלצה מעשית.

{history_text}

מקורות מהמאגר:
{context}

שאלה: {query}

תשובה מפורטת ומועילה:"""
    for attempt in range(3):
        try:
            return client_gemini.models.generate_content(model="gemini-2.5-flash", contents=prompt).text
        except Exception as e:
            if attempt < 2 and ("503" in str(e) or "UNAVAILABLE" in str(e) or "429" in str(e)):
                time.sleep(5)
                continue
            raise

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(content=HTML_PAGE, headers={
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0"
    })

@app.post("/api/search")
async def search(request: Request, query: str = Form(...), history: str = Form("[]")):
    try:
        q=query.strip()
        heb=sum(1 for c in q if 0x05D0<=ord(c)<=0x05EA)
        words=[w for w in q.split() if len(w)>=2]
        NON_LEGAL = re.compile(
            r"^(שלום|היי|הי|בוקר טוב|ערב טוב|לילה טוב|מה שלומך|מה נשמע|תודה|ביי|להתראות|בסדר|אוקי|כן|לא|ok|yes|no|hi|hello|bye|merci|bonjour|bonsoir)[\s!?.]*$",
            re.IGNORECASE
        )
        if NON_LEGAL.match(q) or len(q)<5 or heb<3 or (len(words)<2 and heb<6):
            return JSONResponse({"answer":"אנא הקלד שאלה משפטית ברורה","sources":[]})
        docs, metas = search_docs(query)
        import json as _json
        try:
            hist_data = _json.loads(history) if history else []
        except Exception:
            hist_data = []
        answer = ask_gemini(query, docs, metas, history=hist_data)
        seen=set();sources=[]
        for d,m in zip(docs,metas):
            if m["filename"] not in seen:
                seen.add(m["filename"]);sources.append({"filename":m["filename"],"extrait":d[:3000],"file_id":m.get("file_id","")})
        try:
            user = auth.get_current_user(request)
            if user and user.get("allowed"):
                database.save_search(user["email"], query, answer, sources)
        except Exception:
            pass
        return JSONResponse({"answer": answer, "sources": sources})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...), query: str = Form("")):
    try:
        all_texts = []
        for file in files:
            content = await file.read()
            fh = io.BytesIO(content)
            ext = Path(file.filename).suffix.lower()
            if ext == ".docx":
                doc = Document(fh)
                txt = chr(10).join(p.text for p in doc.paragraphs)
            elif ext == ".pdf":
                try:
                    import PyPDF2
                    reader = PyPDF2.PdfReader(fh)
                    pages_txt = []
                    for p in reader.pages:
                        t = p.extract_text() or ""
                        if t:
                            heb = sum(1 for c in t if 0x05D0 <= ord(c) <= 0x05EA)
                            if len(t) > 0 and heb / len(t) > 0.1:
                                pages_txt.append(t)
                    txt = chr(10).join(pages_txt)
                except Exception as pdf_err:
                    txt = "[PDF illisible: "+str(pdf_err)[:80]+"]"
            else:
                txt = content.decode("utf-8", errors="ignore")
            all_texts.append("["+file.filename+"]:\n"+txt[:2000])
        text = chr(10).join(all_texts)
        docs, metas = search_docs(query if query else text[:500])
        context = chr(10).join("[" + m["filename"] + "]\n" + d for d,m in zip(docs,metas))
        prompt = f"""\u05D0\u05EA\u05D4 \u05E2\u05D5\u05D6\u05E8 \u05DE\u05E9\u05E4\u05D8\u05D9 \u05DE\u05D5\u05DE\u05D7\u05D4. \u05D4\u05DC\u05E7\u05D5\u05D7 \u05D4\u05E2\u05DC\u05D4 \u05DE\u05E1\u05DE\u05DA.
\u05DE\u05E1\u05DE\u05DA \u05D4\u05DC\u05E7\u05D5\u05D7:
{text[:3000]}
\u05DE\u05E7\u05D5\u05E8\u05D5\u05EA \u05E8\u05DC\u05D5\u05D5\u05E0\u05D8\u05D9\u05D9\u05DD:
{context}
{"\u05E9\u05D0\u05DC\u05D4: " + query if query else "\u05EA\u05DF \u05E0\u05D9\u05EA\u05D5\u05D7 \u05DE\u05E9\u05E4\u05D8\u05D9 \u05E9\u05DC \u05D4\u05DE\u05E1\u05DE\u05DA \u05E2\u05DD \u05D4\u05E4\u05E0\u05D9\u05D4 \u05DC\u05DE\u05E7\u05D5\u05E8\u05D5\u05EA."}"""
        response = client_gemini.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        seenU=set();srcU=[]
        for d,m in zip(docs,metas):
            if m["filename"] not in seenU:
                seenU.add(m["filename"]);srcU.append({"filename":m["filename"],"extrait":d[:3000]})
        return JSONResponse({"answer": response.text, "sources": srcU})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

HTML_PAGE = '''
<!DOCTYPE html>
<html dir="rtl" lang="he">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>&#x05DE;&#x05E2;&#x05E8;&#x05DB;&#x05EA; &#x05D1;&#x05D3;&quot;&#x05E8; AI</title><link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>⚖️</text></svg>">
<style>
@import url('https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;600;700;800&display=swap');
:root {
  --gold:#c9a227; --gold-light:#e0b52f; --gold-dim:rgba(201,162,39,0.13);
  --dark:#0a0e14; --dark2:#111722; --dark3:#1a2332; --dark4:#243044;
  --text:#e8e8e8; --text-dim:#8a9bb0; --border:rgba(201,162,39,0.22);
}
body.light {
  --gold:#c08a3e; --gold-light:#d4a55c; --gold-dim:rgba(192,138,62,0.12);
  --dark:#eef1f5; --dark2:#ffffff; --dark3:#ffffff; --dark4:#f7f8fa;
  --text:#1a2433; --text-dim:#5a6675; --border:rgba(15,40,71,0.12);
}
body.light .header { background:linear-gradient(135deg,#0f2847,#1e466f) !important; border-bottom:3px solid #d4a55c; }
body.light .header h1 { color:#ffffff; }
body.light .header p { color:#b9c9dc !important; }
body.light .header-badge { background:rgba(212,165,92,0.18); border-color:rgba(212,165,92,0.5); color:#e3c089; }
body.light .search-box, body.light .answer { background:#ffffff; border-color:#e6e9ef; box-shadow:0 2px 14px rgba(15,40,71,0.06); }
body.light .search-box textarea { background:#f7f8fa; border-color:#e0e4ea; }
body.light .source-item { background:#ffffff; border:1px solid #e6e9ef; border-right:3px solid #d4a55c; color:#5a6675; }
body.light .source-item:hover { background:#f7f8fa; }
body.light .btn { color:#ffffff; }
body.light .answer-title, body.light .sources-title { color:#0f2847; }
body.light .answer-body { color:#1a2433; }
body.light .faq-item { background:#ffffff !important; border-color:#e6e9ef !important; }
body.light .faq-title { color:#0f2847; }
body.light .faq-q { color:#1a2433; }
body.light .faq-q:hover { color:#c08a3e; }
body.light .faq-a { color:#5a6675; }
body.light .modal { background:#ffffff; }
body.light .modal h3 { color:#0f2847; }
body.light .modal p { color:#5a6675; }
body.light .pill { background:#ffffff; border-color:#e0e4ea; color:#5a6675; }
.theme-toggle {
  position:absolute; top:18px; left:18px; background:var(--gold-dim);
  border:1px solid var(--border); color:var(--gold); width:42px; height:42px;
  border-radius:50%; font-size:1.2em; cursor:pointer; z-index:10;
  transition:all 0.2s; display:flex; align-items:center; justify-content:center;
}
.theme-toggle:hover { background:var(--gold); color:var(--dark); }
* { margin:0; padding:0; box-sizing:border-box; }
body { background:var(--dark); color:var(--text); font-family:'Heebo',sans-serif; min-height:100vh; direction:rtl; display:flex; flex-direction:column; }
.page-wrap { flex:1 0 auto; }
.header { background:linear-gradient(135deg,#0d1520,#1a2840); padding:60px 20px 50px; text-align:center; border-bottom:1px solid var(--border); position:relative; overflow:hidden; }
.header::before { content:''; position:absolute; top:-60px; left:50%; transform:translateX(-50%); width:500px; height:400px; background:radial-gradient(circle, rgba(201,162,39,0.07) 0%, transparent 70%); pointer-events:none; }
.header-badge { display:inline-flex; align-items:center; gap:6px; background:var(--gold-dim); border:1px solid var(--border); border-radius:20px; padding:5px 14px; font-size:0.82em; color:var(--gold); margin-bottom:14px; }
.header h1 { color:var(--gold); font-size:2.4em; font-weight:800; letter-spacing:-0.5px; }
.header p { color:var(--text-dim); margin-top:8px; font-size:1em; line-height:1.5; }
.header-stats { display:flex; justify-content:center; gap:36px; margin-top:22px; flex-wrap:wrap; }
.hstat { text-align:center; cursor:pointer; padding:8px 14px; border-radius:10px; transition:all 0.2s; border:1px solid transparent; }
.hstat:hover { background:var(--gold-dim); border-color:var(--border); }
.hstat-num { font-size:1.5em; font-weight:800; color:var(--gold); }
.hstat-label { font-size:0.75em; color:var(--text-dim); margin-top:2px; }
.container { max-width:900px; margin:28px auto; padding:0 20px; }
.search-box { background:var(--dark3); border:1px solid var(--border); border-radius:16px; padding:24px; margin-bottom:20px; }
.search-box textarea { width:100%; padding:14px; border:1px solid #2a3545; border-radius:10px; background:var(--dark2); color:var(--text); font-family:'Heebo',sans-serif; font-size:1.05em; resize:vertical; min-height:85px; direction:rtl; transition:border 0.2s; }
.search-box textarea:focus { outline:none; border-color:var(--gold); box-shadow:0 0 0 3px var(--gold-dim); }
.btn { background:var(--gold); color:var(--dark); border:none; padding:13px 28px; border-radius:10px; font-family:'Heebo',sans-serif; font-size:1em; font-weight:700; cursor:pointer; margin-top:14px; transition:all 0.2s; }
.btn:hover { background:var(--gold-light); transform:translateY(-1px); box-shadow:0 4px 20px rgba(201,162,39,0.3); }
.btn-help { background:transparent; color:var(--text-dim); border:1px solid #2a3545; padding:13px 18px; border-radius:10px; font-family:'Heebo',sans-serif; font-size:0.92em; cursor:pointer; margin-top:14px; margin-right:8px; transition:all 0.2s; }
.btn-help:hover { border-color:var(--gold); color:var(--gold); }
.btn-pdf { background:transparent; border:1px solid var(--border); color:var(--text-dim); padding:7px 14px; border-radius:8px; font-family:'Heebo',sans-serif; font-size:0.83em; cursor:pointer; transition:all 0.2s; }
.btn-pdf:hover { border-color:var(--gold); color:var(--gold); }
.btn-draft { background:transparent; border:1px solid #3498db; color:#3498db; padding:7px 14px; border-radius:8px; font-family:'Heebo',sans-serif; font-size:0.83em; cursor:pointer; margin-right:8px; }
.btn-draft:hover { background:#3498db; color:#fff; }
.upload-area { border:1.5px dashed #2a3545; border-radius:10px; padding:16px; text-align:center; margin-top:12px; cursor:pointer; color:var(--text-dim); font-size:0.9em; transition:all 0.2s; }
.upload-area:hover { border-color:var(--gold); color:var(--gold); background:var(--gold-dim); }
.answer { background:var(--dark3); border:1px solid var(--border); border-radius:16px; padding:24px; margin-top:18px; display:none; }
.answer-header { display:flex; align-items:center; justify-content:space-between; margin-bottom:16px; padding-bottom:14px; border-bottom:1px solid var(--border); }
.answer-title { color:var(--gold); font-size:1em; font-weight:600; }
.answer-body { line-height:1.95; color:var(--text); white-space:pre-wrap; font-size:1.25em; }
.disclaimer { background:rgba(201,162,39,0.05); border:1px solid rgba(201,162,39,0.18); border-radius:10px; padding:12px 16px; margin-top:18px; font-size:0.8em; color:var(--text-dim); line-height:1.5; }
.sources { margin-top:20px; }
.sources-title { font-size:0.88em; font-weight:600; color:var(--gold); margin-bottom:10px; }
.source-item { background:var(--dark2); border-right:3px solid var(--gold); padding:13px 16px; margin-top:8px; border-radius:8px; font-size:1.02em; color:var(--text-dim); transition:all 0.2s; }
.source-item:hover { background:var(--dark4); color:var(--text); }
.loading { text-align:center; padding:36px 20px; display:none; }
.spinner { width:38px; height:38px; border:3px solid var(--dark4); border-top:3px solid var(--gold); border-radius:50%; animation:spin 0.8s linear infinite; margin:0 auto 12px; }
@keyframes spin { to { transform:rotate(360deg); } }
.loading p { color:var(--text-dim); }
.modal-overlay { position:fixed; inset:0; background:rgba(0,0,0,0.72); display:flex; align-items:center; justify-content:center; z-index:999; opacity:0; pointer-events:none; transition:opacity 0.2s; padding:20px; }
.modal-overlay.open { opacity:1; pointer-events:all; }
.modal { background:var(--dark3); border:1px solid var(--border); border-radius:16px; padding:28px; max-width:440px; width:100%; transform:scale(0.95); transition:transform 0.2s; direction:rtl; }
.modal-overlay.open .modal { transform:scale(1); }
.modal h3 { color:var(--gold); font-size:1.1em; margin-bottom:10px; }
.modal p { color:var(--text-dim); line-height:1.7; font-size:0.92em; }
.modal-close { display:inline-block; margin-top:18px; background:var(--gold); color:var(--dark); border:none; padding:9px 22px; border-radius:8px; font-family:'Heebo',sans-serif; font-weight:700; cursor:pointer; }
footer { background:var(--dark3); border-top:1px solid var(--border); padding:22px 20px; margin-top:auto; flex-shrink:0; }
.footer-inner { max-width:900px; margin:0 auto; display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px; }
.footer-logo { color:var(--gold); font-weight:700; font-size:1em; }
.footer-text { color:var(--text-dim); font-size:0.8em; text-align:center; }
.footer-link { color:var(--gold); text-decoration:none; }
.footer-link:hover { color:var(--gold-light); }
.footer-tech { color:var(--text-dim); font-size:0.75em; }
.btn-new-chat { background:linear-gradient(135deg,#1a5c3a,#0f3d26); border:1px solid #2ecc71; color:#2ecc71; padding:13px 18px; border-radius:10px; font-family:"Heebo",sans-serif; font-size:0.92em; cursor:pointer; margin-top:14px; margin-left:8px; transition:all 0.2s; }
.btn-new-chat:hover { background:linear-gradient(135deg,#2ecc71,#27ae60); color:#0a0e14; font-weight:700; }
.chat-history-indicator { display:none; background:rgba(46,204,113,0.1); border:1px solid rgba(46,204,113,0.3); border-radius:8px; padding:8px 14px; margin-bottom:10px; font-size:0.85em; color:#2ecc71; }
.btn-reset-search { background:rgba(192,57,43,0.08); border:1px solid rgba(231,76,60,0.4); color:#e07a6f; padding:13px 18px; border-radius:10px; font-family:'Heebo',sans-serif; font-size:0.9em; cursor:pointer; margin-top:14px; float:left; transition:all 0.2s; }
.btn-reset-search:hover { background:rgba(192,57,43,0.18); border-color:#e74c3c; color:#ff6b5b; }
.search-box::after { content:''; display:table; clear:both; }
.howto-section { max-width:900px; margin:50px auto 0; padding:0 20px; }
.howto-title { text-align:center; color:var(--gold); font-size:1.5em; font-weight:800; margin-bottom:6px; }
.howto-sub { text-align:center; color:var(--text-dim); font-size:0.9em; margin-bottom:30px; }
.howto-steps { display:flex; gap:18px; flex-wrap:wrap; justify-content:center; }
.howto-step { flex:1; min-width:220px; background:var(--dark3); border:1px solid var(--border); border-radius:14px; padding:26px 22px; text-align:center; position:relative; transition:all 0.25s; }
.howto-step:hover { border-color:var(--gold); transform:translateY(-3px); box-shadow:0 8px 24px rgba(0,0,0,0.3); }
.howto-num { width:44px; height:44px; margin:0 auto 14px; background:linear-gradient(135deg,var(--gold),#8b6914); border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:1.3em; font-weight:800; color:#0a0e14; }
.howto-step h4 { color:var(--text); font-size:1.05em; font-weight:700; margin-bottom:8px; }
.howto-step p { color:var(--text-dim); font-size:0.88em; line-height:1.6; }
.faq-section { max-width:780px; margin:55px auto 0; padding:0 20px; }
.faq-title { text-align:center; color:var(--gold); font-size:1.5em; font-weight:800; margin-bottom:6px; }
.faq-sub { text-align:center; color:var(--text-dim); font-size:0.9em; margin-bottom:28px; }
.faq-item { background:linear-gradient(135deg,#1c2636,#16202e); border:1px solid var(--border); border-radius:14px; margin-bottom:12px; overflow:hidden; transition:all 0.25s; box-shadow:0 2px 10px rgba(0,0,0,0.15); }
.faq-item:hover { border-color:rgba(201,162,39,0.4); }
.faq-item.open { border-color:var(--gold); box-shadow:0 4px 18px rgba(201,162,39,0.12); }
.faq-q { padding:18px 22px; cursor:pointer; display:flex; align-items:center; justify-content:space-between; font-weight:600; color:var(--text); font-size:1em; transition:color 0.2s; user-select:none; gap:12px; }
.faq-q:hover { color:var(--gold); }
.faq-q .qmark { width:26px; height:26px; flex-shrink:0; background:var(--gold-dim); border:1px solid var(--border); border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:0.8em; color:var(--gold); font-weight:700; }
.faq-q .qtext { flex:1; }
.faq-q .arrow { color:var(--gold); transition:transform 0.25s; font-size:0.8em; flex-shrink:0; }
.faq-item.open .faq-q .arrow { transform:rotate(180deg); }
.faq-a { max-height:0; overflow:hidden; transition:max-height 0.3s ease, padding 0.3s ease; padding:0 22px 0 60px; color:var(--text-dim); line-height:1.85; font-size:0.93em; }
.faq-item.open .faq-a { max-height:320px; padding:0 22px 20px 60px; }
.cta-nav { color:#0a0e14 !important; background:linear-gradient(135deg,#e0b52f,#c9a227) !important; font-weight:700 !important; box-shadow:0 2px 12px rgba(201,162,39,0.35); }
.cta-nav:hover { background:linear-gradient(135deg,#f0c640,#d4ae2f) !important; transform:translateY(-1px); }
.filters-wrap { margin-bottom:16px; }
.filters-head { display:flex; align-items:center; justify-content:space-between; margin-bottom:10px; flex-wrap:wrap; gap:8px; }
.filters-label { color:var(--text-dim); font-size:0.86em; display:flex; align-items:center; gap:6px; }
.filters-count { color:var(--gold); font-size:0.8em; font-weight:600; }
.filters-pills { display:flex; gap:9px; flex-wrap:wrap; }
.pill { padding:8px 16px; border:1.5px solid #2a3545; border-radius:22px; background:var(--dark2); color:var(--text-dim); font-family:'Heebo',sans-serif; font-size:0.88em; font-weight:500; cursor:pointer; transition:all 0.18s; display:flex; align-items:center; gap:6px; user-select:none; }
.pill:hover { border-color:var(--gold); color:var(--gold); }
.pill.on { background:linear-gradient(135deg,var(--gold),#b8941f); border-color:var(--gold); color:#0a0e14; font-weight:700; box-shadow:0 2px 10px rgba(201,162,39,0.3); }
.pill .chk { font-size:0.85em; }
.btn-clear { background:transparent; border:none; color:var(--text-dim); font-family:'Heebo',sans-serif; font-size:0.8em; cursor:pointer; text-decoration:underline; padding:0; transition:color 0.2s; }
.btn-clear:hover { color:var(--gold); }
.topnav { background:rgba(10,14,20,0.97); border-bottom:1px solid var(--border); padding:0 32px; display:flex; align-items:center; justify-content:space-between; height:60px; position:sticky; top:0; z-index:100; }
.topnav-logo { display:flex; align-items:center; gap:9px; text-decoration:none; cursor:pointer; }
.topnav-logo .ico { width:34px; height:34px; background:linear-gradient(135deg,var(--gold),#8b6914); border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:1.1em; }
.topnav-logo .txt { font-size:1.05em; font-weight:700; color:var(--gold); }
.topnav-links { display:flex; align-items:center; gap:6px; }
.topnav-links a { color:var(--text-dim); text-decoration:none; padding:8px 16px; border-radius:8px; font-size:0.93em; font-weight:500; transition:all 0.2s; }
.topnav-links a:hover { color:var(--gold); background:var(--gold-dim); }
.topnav-links a.cta { color:var(--dark); background:var(--gold); font-weight:600; }
.topnav-links a.cta:hover { background:var(--gold-light); }
@media(max-width:600px){ .topnav { padding:0 16px; } .topnav-links a { padding:7px 11px; font-size:0.85em; } }
@media(max-width:600px) { .header h1 { font-size:1.7em; } .header-stats { gap:16px; } .footer-inner { flex-direction:column; text-align:center; } }
.header h1,.header p,.header-badge,.header-stats{position:relative;z-index:2;transition:transform 0.1s linear;}</style>
</head>
<body>

<nav class="topnav">
  <a class="topnav-logo" href="https://law-mor.com" target="_blank">
    <span class="ico">&#x2696;&#xFE0F;</span>
    <span class="txt">&#x05D1;&#x05D3;&quot;&#x05E8; AI</span>
  </a>
  <div class="topnav-links">
  </div>
</nav>

<div class="page-wrap">

<div class="modal-overlay" id="modal-overlay">
  <div class="modal">
    <h3 id="modal-title"></h3>
    <p id="modal-body"></p>
    <button class="modal-close" id="modal-close-btn">&#x05D4;&#x05D1;&#x05E0;&#x05EA;&#x05D9; &#x2714;</button>
  </div>
</div>

<div class="header">
  <button class="theme-toggle" onclick="toggleTheme()" title="מצב תצוגה">☀️</button>
  <div class="header-badge">&#x2728; &#x05DE;&#x05E2;&#x05E8;&#x05DB;&#x05EA; AI &#x05DC;&#x05D7;&#x05D9;&#x05E4;&#x05D5;&#x05E9; &#x05E4;&#x05E1;&#x05D9;&#x05E7;&#x05D4; &#x05E8;&#x05D1;&#x05E0;&#x05D9;&#x05EA;</div>
  <h1>&#x2696;&#xFE0F; &#x05DE;&#x05E2;&#x05E8;&#x05DB;&#x05EA; &#x05D1;&#x05D3;&quot;&#x05E8; AI</h1>
  <p>&#x05DE;&#x05E2;&#x05E8;&#x05DB;&#x05EA; &#x05E2;&#x05D6;&#x05E8; &#x05DC;&#x05D9;&#x05D9;&#x05E6;&#x05D5;&#x05D2; &#x05DE;&#x05E9;&#x05E4;&#x05D8;&#x05D9; &#x05D5;&#x05EA;&#x05D5;&#x05E8;&#x05E0;&#x05D9; &#x05D1;&#x05D1;&#x05EA;&#x05D9; &#x05D4;&#x05D3;&#x05D9;&#x05DF; &#x05D4;&#x05E8;&#x05D1;&#x05E0;&#x05D9;&#x05D9;&#x05DD; &#x05D1;&#x05D9;&#x05E9;&#x05E8;&#x05D0;&#x05DC;</p>
  <div class="header-stats">
    <div class="hstat" id="stat1">
      <div class="hstat-num">63K+</div>
      <div class="hstat-label">&#x05DE;&#x05E7;&#x05D5;&#x05E8;&#x05D5;&#x05EA; &#x05D1;&#x05D0;&#x05D9;&#x05E0;&#x05D3;&#x05E7;&#x05E1; &#x2139;</div>
    </div>
    <div class="hstat" id="stat2">
      <div class="hstat-num">AI</div>
      <div class="hstat-label">&#x05DE;&#x05E0;&#x05D5;&#x05E2; AI &#x2139;</div>
    </div>
    <div class="hstat" id="stat3">
      <div class="hstat-num">3,283</div>
      <div class="hstat-label">&#x05E4;&#x05E1;&#x05E7;&#x05D9; &#x05D3;&#x05D9;&#x05DF; &#x2139;</div>
    </div>
  </div>
</div>

<div class="container">
<div class="search-box" id="search-zone">
<div class="filters-wrap">
  <div class="filters-head">
    <span class="filters-label">&#x1F3F7;&#xFE0F; &#x05E1;&#x05E0;&#x05DF; &#x05DC;&#x05E4;&#x05D9; &#x05E0;&#x05D5;&#x05E9;&#x05D0; (&#x05E0;&#x05D9;&#x05EA;&#x05DF; &#x05DC;&#x05D1;&#x05D7;&#x05D5;&#x05E8; &#x05DB;&#x05DE;&#x05D4;)</span>
    <span class="filters-count" id="filters-count"></span>
  </div>
  <div class="filters-pills" id="filters-pills"></div>
</div>
<div class="chat-history-indicator" id="chat-indicator">&#x1F4AC; <span id="chat-count">0</span> &#x05E9;&#x05D0;&#x05DC;&#x05D5;&#x05EA; &#x05D1;&#x05D4;&#x05E7;&#x05E9;&#x05E8; &mdash; &#x05DC;&#x05E9;&#x05D0;&#x05DC;&#x05D4; &#x05D7;&#x05D3;&#x05E9;&#x05D4; &#x05DC;&#x05D7;&#x05E5; &#x05E2;&#x05DC; &#x05D4;&#x05DB;&#x05E4;&#x05EA;&#x05D5;&#x05E8; &#x05D4;&#x05D9;&#x05E8;&#x05D5;&#x05E7;</div><textarea id="query" placeholder="&#x05E9;&#x05D0;&#x05DC;&#x05D4; 1: &#x05D4;&#x05E7;&#x05DC;&#x05D3; &#x05D0;&#x05EA; &#x05E9;&#x05D0;&#x05DC;&#x05EA;&#x05DA; &#x05D4;&#x05DE;&#x05E9;&#x05E4;&#x05D8;&#x05D9;&#x05EA; &#x05DB;&#x05D0;&#x05DF;..."></textarea>
<div class="upload-area" onclick="document.getElementById('file').click()">
&#x1F4CE; &#x05D4;&#x05E2;&#x05DC;&#x05D4; &#x05DE;&#x05E1;&#x05DE;&#x05DA; (&#x05D0;&#x05D5;&#x05E4;&#x05E6;&#x05D9;&#x05D5;&#x05E0;&#x05DC;&#x05D9;) - <span id="filename">&#x05DC;&#x05D0; &#x05E0;&#x05D1;&#x05D7;&#x05E8; &#x05E7;&#x05D5;&#x05D1;&#x05E5;</span>
<input type="file" id="file" style="display:none" accept=".docx,.pdf,.txt" multiple onchange="showFiles(this)">
</div>
<div id="filelist" style="margin-top:10px;"></div>
<button class="btn" onclick="search()">&#x1F50D; &#x05D7;&#x05E4;&#x05E9; &#x05D1;&#x05DE;&#x05D0;&#x05D2;&#x05E8;</button>
<button class="btn-help" id="help-btn">&#x1F4A1; &#x05DB;&#x05D9;&#x05E6;&#x05D3; &#x05DC;&#x05D7;&#x05E4;&#x05E9;?</button><button class="btn-reset-search" id="reset-search-btn">&#x1F5D1;&#xFE0F; &#x05E0;&#x05E7;&#x05D4; &#x05D4;&#x05DB;&#x05DC;</button>
</div>
<div class="loading" id="loading">
  <div class="spinner"></div>
  <p>&#x05DE;&#x05D7;&#x05E4;&#x05E9; &#x05D1;&#x05DE;&#x05D0;&#x05D2;&#x05E8; &#x05E4;&#x05E1;&#x05E7;&#x05D9; &#x05D4;&#x05D3;&#x05D9;&#x05DF;...</p>
</div>
<div id="conversation" style="display:none;margin-top:18px;"></div>
</div>

</div>

</div>

<div class="howto-section">
  <div class="howto-title">&#x05D0;&#x05D9;&#x05DA; &#x05D6;&#x05D4; &#x05E2;&#x05D5;&#x05D1;&#x05D3;?</div>
  <div class="howto-sub">&#x05E9;&#x05DC;&#x05D5;&#x05E9;&#x05D4; &#x05E9;&#x05DC;&#x05D1;&#x05D9;&#x05DD; &#x05E4;&#x05E9;&#x05D5;&#x05D8;&#x05D9;&#x05DD; &#x05DC;&#x05E7;&#x05D1;&#x05DC;&#x05EA; &#x05EA;&#x05E9;&#x05D5;&#x05D1;&#x05D4; &#x05DE;&#x05E9;&#x05E4;&#x05D8;&#x05D9;&#x05EA;</div>
  <div class="howto-steps">
    <div class="howto-step">
      <div class="howto-num">1</div>
      <h4>&#x05D4;&#x05E7;&#x05DC;&#x05D3; &#x05E9;&#x05D0;&#x05DC;&#x05D4;</h4>
      <p>&#x05D4;&#x05E7;&#x05DC;&#x05D3; &#x05E9;&#x05D0;&#x05DC;&#x05D4; &#x05DE;&#x05E9;&#x05E4;&#x05D8;&#x05D9;&#x05EA; &#x05D1;&#x05E2;&#x05D1;&#x05E8;&#x05D9;&#x05EA; &#x05D0;&#x05D5; &#x05D1;&#x05D7;&#x05E8; &#x05E0;&#x05D5;&#x05E9;&#x05D0; &#x05DE;&#x05D4;&#x05E4;&#x05D9;&#x05DC;&#x05D8;&#x05E8;&#x05D9;&#x05DD;</p>
    </div>
    <div class="howto-step">
      <div class="howto-num">2</div>
      <h4>&#x05D4;&#x05DE;&#x05E2;&#x05E8;&#x05DB;&#x05EA; &#x05DE;&#x05D7;&#x05E4;&#x05E9;&#x05EA;</h4>
      <p>&#x05D4;&#x05DE;&#x05E2;&#x05E8;&#x05DB;&#x05EA; &#x05E1;&#x05D5;&#x05E8;&#x05E7;&#x05EA; &#x05D0;&#x05EA; &#x05DE;&#x05D0;&#x05D2;&#x05E8; &#x05E4;&#x05E1;&#x05E7;&#x05D9; &#x05D4;&#x05D3;&#x05D9;&#x05DF; &#x05D5;&#x05DE;&#x05D0;&#x05EA;&#x05E8;&#x05EA; &#x05D0;&#x05EA; &#x05D4;&#x05DE;&#x05E7;&#x05D5;&#x05E8;&#x05D5;&#x05EA; &#x05D4;&#x05E8;&#x05DC;&#x05D5;&#x05D5;&#x05E0;&#x05D8;&#x05D9;&#x05D9;&#x05DD;</p>
    </div>
    <div class="howto-step">
      <div class="howto-num">3</div>
      <h4>&#x05E7;&#x05D1;&#x05DC; &#x05EA;&#x05E9;&#x05D5;&#x05D1;&#x05D4;</h4>
      <p>&#x05E7;&#x05D1;&#x05DC; &#x05EA;&#x05E9;&#x05D5;&#x05D1;&#x05D4; &#x05DE;&#x05E4;&#x05D5;&#x05E8;&#x05D8;&#x05EA; &#x05E2;&#x05DD; &#x05D4;&#x05E4;&#x05E0;&#x05D9;&#x05D4; &#x05DC;&#x05DE;&#x05E7;&#x05D5;&#x05E8;&#x05D5;&#x05EA; &#x2014; &#x05E0;&#x05D9;&#x05EA;&#x05DF; &#x05DC;&#x05D9;&#x05D9;&#x05E6;&#x05D0; &#x05DC;-PDF</p>
    </div>
  </div>
</div>

<div class="faq-section">
  <div class="faq-title">&#x05E9;&#x05D0;&#x05DC;&#x05D5;&#x05EA; &#x05E0;&#x05E4;&#x05D5;&#x05E6;&#x05D5;&#x05EA;</div>
  <div class="faq-sub">&#x05DB;&#x05DC; &#x05DE;&#x05D4; &#x05E9;&#x05E8;&#x05E6;&#x05D9;&#x05EA;&#x05DD; &#x05DC;&#x05D3;&#x05E2;&#x05EA; &#x05E2;&#x05DC; &#x05D4;&#x05DE;&#x05E2;&#x05E8;&#x05DB;&#x05EA;</div>
  <div class="faq-item">
    <div class="faq-q"><span class="qmark">?</span><span class="qtext">&#x05DE;&#x05D4;&#x05D9; &#x05DE;&#x05E2;&#x05E8;&#x05DB;&#x05EA; &#x05D1;&#x05D3;&quot;&#x05E8; AI?</span><span class="arrow">&#x25BC;</span></div>
    <div class="faq-a">&#x05DE;&#x05E2;&#x05E8;&#x05DB;&#x05EA; &#x05D1;&#x05D3;&quot;&#x05E8; AI &#x05D4;&#x05D9;&#x05D0; &#x05DB;&#x05DC;&#x05D9; &#x05E2;&#x05D6;&#x05E8; &#x05DE;&#x05EA;&#x05E7;&#x05D3;&#x05DD; &#x05DC;&#x05D7;&#x05D9;&#x05E4;&#x05D5;&#x05E9; &#x05D5;&#x05E0;&#x05D9;&#x05EA;&#x05D5;&#x05D7; &#x05E4;&#x05E1;&#x05D9;&#x05E7;&#x05D4; &#x05E8;&#x05D1;&#x05E0;&#x05D9;&#x05EA;. &#x05D4;&#x05DE;&#x05E2;&#x05E8;&#x05DB;&#x05EA; &#x05DE;&#x05D0;&#x05E4;&#x05E9;&#x05E8;&#x05EA; &#x05DC;&#x05D7;&#x05E4;&#x05E9; &#x05D1;&#x05E9;&#x05E4;&#x05D4; &#x05D7;&#x05D5;&#x05E4;&#x05E9;&#x05D9;&#x05EA; &#x05D1;&#x05DE;&#x05D0;&#x05D2;&#x05E8; &#x05E4;&#x05E1;&#x05E7;&#x05D9; &#x05D3;&#x05D9;&#x05DF; &#x05D5;&#x05DC;&#x05E7;&#x05D1;&#x05DC; &#x05E0;&#x05D9;&#x05EA;&#x05D5;&#x05D7; &#x05DE;&#x05E9;&#x05E4;&#x05D8;&#x05D9; &#x05DE;&#x05D1;&#x05D5;&#x05E1;&#x05E1; &#x05DE;&#x05E7;&#x05D5;&#x05E8;&#x05D5;&#x05EA;.</div>
  </div>
  <div class="faq-item">
    <div class="faq-q"><span class="qmark">?</span><span class="qtext">&#x05E2;&#x05DC; &#x05DE;&#x05D4; &#x05DE;&#x05D1;&#x05D5;&#x05E1;&#x05E1;&#x05D5;&#x05EA; &#x05D4;&#x05EA;&#x05E9;&#x05D5;&#x05D1;&#x05D5;&#x05EA;?</span><span class="arrow">&#x25BC;</span></div>
    <div class="faq-a">&#x05D4;&#x05EA;&#x05E9;&#x05D5;&#x05D1;&#x05D5;&#x05EA; &#x05DE;&#x05D1;&#x05D5;&#x05E1;&#x05E1;&#x05D5;&#x05EA; &#x05E2;&#x05DC; &#x05D0;&#x05D9;&#x05E0;&#x05D3;&#x05E7;&#x05E1; &#x05E9;&#x05DC; 3,283 &#x05E4;&#x05E1;&#x05E7;&#x05D9; &#x05D3;&#x05D9;&#x05DF; &#x05DE;&#x05D1;&#x05EA;&#x05D9; &#x05D4;&#x05D3;&#x05D9;&#x05DF; &#x05D4;&#x05E8;&#x05D1;&#x05E0;&#x05D9;&#x05D9;&#x05DD; &#x05D1;&#x05D9;&#x05E9;&#x05E8;&#x05D0;&#x05DC;. &#x05DB;&#x05DC; &#x05EA;&#x05E9;&#x05D5;&#x05D1;&#x05D4; &#x05DE;&#x05E6;&#x05D9;&#x05D2;&#x05D4; &#x05D0;&#x05EA; &#x05D4;&#x05DE;&#x05E7;&#x05D5;&#x05E8;&#x05D5;&#x05EA; &#x05E9;&#x05E2;&#x05DC;&#x05D9;&#x05D4;&#x05DD; &#x05D4;&#x05D9;&#x05D0; &#x05DE;&#x05E1;&#x05EA;&#x05DE;&#x05DB;&#x05EA;.</div>
  </div>
  <div class="faq-item">
    <div class="faq-q"><span class="qmark">?</span><span class="qtext">&#x05D4;&#x05D0;&#x05DD; &#x05D4;&#x05EA;&#x05E9;&#x05D5;&#x05D1;&#x05D5;&#x05EA; &#x05DE;&#x05D4;&#x05D5;&#x05D5;&#x05EA; &#x05D9;&#x05D9;&#x05E2;&#x05D5;&#x05E5; &#x05DE;&#x05E9;&#x05E4;&#x05D8;&#x05D9;?</span><span class="arrow">&#x25BC;</span></div>
    <div class="faq-a">&#x05DC;&#x05D0;. &#x05D4;&#x05DE;&#x05E2;&#x05E8;&#x05DB;&#x05EA; &#x05D4;&#x05D9;&#x05D0; &#x05DB;&#x05DC;&#x05D9; &#x05E2;&#x05D6;&#x05E8; &#x05D1;&#x05DC;&#x05D1;&#x05D3; &#x05D5;&#x05D0;&#x05D9;&#x05E0;&#x05D4; &#x05DE;&#x05D4;&#x05D5;&#x05D5;&#x05D4; &#x05D9;&#x05D9;&#x05E2;&#x05D5;&#x05E5; &#x05DE;&#x05E9;&#x05E4;&#x05D8;&#x05D9;. &#x05D9;&#x05E9; &#x05DC;&#x05D4;&#x05EA;&#x05D9;&#x05D9;&#x05E2;&#x05E5; &#x05E2;&#x05DD; &#x05E2;&#x05D5;&#x05E8;&#x05DA; &#x05D3;&#x05D9;&#x05DF; &#x05D0;&#x05D5; &#x05D8;&#x05D5;&#x05E2;&#x05DF; &#x05E8;&#x05D1;&#x05E0;&#x05D9; &#x05DE;&#x05D5;&#x05E1;&#x05DE;&#x05DA; &#x05DC;&#x05E4;&#x05E0;&#x05D9; &#x05E7;&#x05D1;&#x05DC;&#x05EA; &#x05D4;&#x05D7;&#x05DC;&#x05D8;&#x05D5;&#x05EA; &#x05DE;&#x05E9;&#x05E4;&#x05D8;&#x05D9;&#x05D5;&#x05EA;.</div>
  </div>
  <div class="faq-item">
    <div class="faq-q"><span class="qmark">?</span><span class="qtext">&#x05DB;&#x05D9;&#x05E6;&#x05D3; &#x05DC;&#x05E0;&#x05E1;&#x05D7; &#x05E9;&#x05D0;&#x05DC;&#x05D4; &#x05D8;&#x05D5;&#x05D1;&#x05D4;?</span><span class="arrow">&#x25BC;</span></div>
    <div class="faq-a">&#x05DB;&#x05D3;&#x05D0;&#x05D9; &#x05DC;&#x05E0;&#x05E1;&#x05D7; &#x05E9;&#x05D0;&#x05DC;&#x05D4; &#x05DE;&#x05E4;&#x05D5;&#x05E8;&#x05D8;&#x05EA; &#x05D5;&#x05D1;&#x05E8;&#x05D5;&#x05E8;&#x05D4;. &#x05E0;&#x05D9;&#x05EA;&#x05DF; &#x05D2;&#x05DD; &#x05DC;&#x05D1;&#x05D7;&#x05D5;&#x05E8; &#x05E0;&#x05D5;&#x05E9;&#x05D0;&#x05D9;&#x05DD; &#x05DE;&#x05D4;&#x05E4;&#x05D9;&#x05DC;&#x05D8;&#x05E8;&#x05D9;&#x05DD; &#x05DB;&#x05D3;&#x05D9; &#x05DC;&#x05DE;&#x05E7;&#x05D3; &#x05D0;&#x05EA; &#x05D4;&#x05D7;&#x05D9;&#x05E4;&#x05D5;&#x05E9;. &#x05DB;&#x05DB;&#x05DC; &#x05E9;&#x05D4;&#x05E9;&#x05D0;&#x05DC;&#x05D4; &#x05DE;&#x05E4;&#x05D5;&#x05E8;&#x05D8;&#x05EA; &#x05D9;&#x05D5;&#x05EA;&#x05E8; &#x2014; &#x05D4;&#x05EA;&#x05E9;&#x05D5;&#x05D1;&#x05D4; &#x05DE;&#x05D3;&#x05D5;&#x05D9;&#x05E7;&#x05EA; &#x05D9;&#x05D5;&#x05EA;&#x05E8;.</div>
  </div>
  <div class="faq-item">
    <div class="faq-q"><span class="qmark">?</span><span class="qtext">&#x05D4;&#x05D0;&#x05DD; &#x05E0;&#x05D9;&#x05EA;&#x05DF; &#x05DC;&#x05D4;&#x05E2;&#x05DC;&#x05D5;&#x05EA; &#x05DE;&#x05E1;&#x05DE;&#x05DA; &#x05DC;&#x05E0;&#x05D9;&#x05EA;&#x05D5;&#x05D7;?</span><span class="arrow">&#x25BC;</span></div>
    <div class="faq-a">&#x05DB;&#x05DF;. &#x05E0;&#x05D9;&#x05EA;&#x05DF; &#x05DC;&#x05D4;&#x05E2;&#x05DC;&#x05D5;&#x05EA; &#x05DE;&#x05E1;&#x05DE;&#x05DA; &#x05D1;&#x05E4;&#x05D5;&#x05E8;&#x05DE;&#x05D8; DOCX, PDF &#x05D0;&#x05D5; TXT &#x05DC;&#x05E0;&#x05D9;&#x05EA;&#x05D5;&#x05D7;. &#x05D4;&#x05DE;&#x05E2;&#x05E8;&#x05DB;&#x05EA; &#x05EA;&#x05E0;&#x05EA;&#x05D7; &#x05D0;&#x05EA; &#x05D4;&#x05DE;&#x05E1;&#x05DE;&#x05DA; &#x05D5;&#x05EA;&#x05E9;&#x05D5;&#x05D5;&#x05D4; &#x05D0;&#x05D5;&#x05EA;&#x05D5; &#x05DC;&#x05DE;&#x05E7;&#x05D5;&#x05E8;&#x05D5;&#x05EA; &#x05D4;&#x05E8;&#x05DC;&#x05D5;&#x05D5;&#x05E0;&#x05D8;&#x05D9;&#x05D9;&#x05DD; &#x05D1;&#x05DE;&#x05D0;&#x05D2;&#x05E8;.</div>
  </div>
</div>

<footer>
  <div class="footer-inner">
    <div class="footer-logo">&#x2696;&#xFE0F; &#x05DE;&#x05E2;&#x05E8;&#x05DB;&#x05EA; &#x05D1;&#x05D3;&quot;&#x05E8; AI</div>
    <div class="footer-text">
      &#x00A9; 2026 <a href="https://law-mor.com" target="_blank" class="footer-link">&#x05DE;&#x05E9;&#x05E8;&#x05D3; &#x05E2;&#x05D5;&quot;&#x05D3; &#x05D0;&#x05D4;&#x05D5;&#x05D3; &#x05E0;&#x05EA;&#x05DF; &#x05DE;&#x05D5;&#x05E8;</a><br>
      <a href="https://law-mor.com/%d7%a6%d7%95%d7%a8-%d7%a7%d7%a9%d7%a8/" target="_blank" class="footer-link">&#x05E6;&#x05D5;&#x05E8; &#x05E7;&#x05E9;&#x05E8;</a> | office@law-mor.com
    </div>
    <div class="footer-tech" style="text-align:right;"><span style="color:var(--text-dim);font-size:0.85em;opacity:0.8;">v1.2.0 <div class="footer-tech"><a href="https://law-mor.com"bull; June 2026</span><br><a href="https://law-mor.com" target="_blank" class="footer-link">&#x05DC;&#x05D0;&#x05EA;&#x05E8; &#x05D4;&#x05DE;&#x05E9;&#x05E8;&#x05D3; &#x05E2;&#x05D5;&quot;&#x05D3; &#x05DE;&#x05D5;&#x05E8; &#x2190;</a></div>
  </div>
</footer>

<script>
var modalTexts = {
  stat1: ["מאגר פסקי הדין", "המערכת כוללת למעלה מ-63,000 קטעים מתוך 3,283 פסקי דין מבתי הדין הרבניים בישראל. הנושאים כוללים גירושין, מזונות, כתובה, משמורת ועוד."],
  stat2: ["Gemini 2.5 Flash", "המערכת משתמשת ב-Gemini 2.5 Flash של Google — מודל מתקדם לעיבוד שפה עברית, המותאם לניתוח טקסטים משפטיים ותורניים."],
  stat3: ["3,283 פסקי דין", "האינדקס מכיל 3,283 פסקי דין מקוריים מבתי הדין הרבניים בישראל, המכסים מגוון רחב של נושאים בדיני משפחה."],
  help: ["כיצד לחפש?", "הקלד שאלה משפטית בעברית. המערכת תמצא פסקי דין רלוונטיים ותנתח אותם בבינה מלאכותית. ניתן גם לצרף מסמך לניתוח. ככל שהשאלה מפורטת יותר — התשובה תהיה מדויקת יותר."]
};

function openModal(title, body) {
  document.getElementById('modal-title').textContent = title;
  document.getElementById('modal-body').textContent = body;
  document.getElementById('modal-overlay').classList.add('open');
}
document.getElementById('modal-close-btn').addEventListener('click', function() { document.getElementById('modal-overlay').classList.remove('open'); });
document.getElementById('modal-overlay').addEventListener('click', function(e) { if (e.target === this) this.classList.remove('open'); });
document.getElementById('stat1').addEventListener('click', function() { openModal(modalTexts.stat1[0], modalTexts.stat1[1]); });
document.getElementById('stat2').addEventListener('click', function() { openModal(modalTexts.stat2[0], modalTexts.stat2[1]); });
document.getElementById('stat3').addEventListener('click', function() { openModal(modalTexts.stat3[0], modalTexts.stat3[1]); });
document.getElementById('help-btn').addEventListener('click', function() { openModal(modalTexts.help[0], modalTexts.help[1]); });

async function generateDraft(btn) {
  var turnDiv = btn.closest ? btn.closest('.answer-body') : null;
  if (!turnDiv) turnDiv = btn.parentElement.parentElement;
  var answerBody = turnDiv.querySelector ? turnDiv.querySelector('.answer-body') : turnDiv;
  var answerText = answerBody ? answerBody.innerText : '';
  var origText = btn.textContent;
  btn.textContent = 'יוצר...';
  btn.disabled = true;
  try {
    var resp = await fetch('/api/draft', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({answer: answerText, query: ''})});
    var data = await resp.json();
    var d = document.createElement('div');
    d.style.cssText = 'margin-top:14px;background:var(--dark2);border:1px solid #3498db;border-radius:8px;padding:14px;direction:rtl;';
    d.innerHTML = '<div style="color:#3498db;font-weight:700;margin-bottom:8px;">טיוטה משפטית</div><div style="white-space:pre-wrap;line-height:1.8;">'+(data.draft||'')+'</div>';
    turnDiv.parentElement ? turnDiv.parentElement.appendChild(d) : document.getElementById('conversation').appendChild(d);
    btn.textContent = 'נוצר';
  } catch(e) { btn.textContent = 'שגיאה'; }
  btn.disabled = false;
}
function exportPDF() {
  var bodyEl = document.querySelector('.answer-body');
  var answerText = bodyEl ? bodyEl.innerText : '';
  var query = document.getElementById('query').value;
  var sources = Array.from(document.querySelectorAll('.source-item')).map(function(s) { return s.textContent.trim(); }).join(String.fromCharCode(10));
  var sep = '================================================';
  var nl=String.fromCharCode(10);var content="BADR AI"+nl+sep+nl+nl+query+nl+nl+sep+nl+nl+answerText+nl+nl+sep+nl+nl+sources+nl+nl+sep+nl+new Date().toLocaleDateString();
  var blob = new Blob(['\uFEFF' + content], {type:'text/plain;charset=utf-8'});
  var url = URL.createObjectURL(blob);
  var a = document.createElement('a');
  a.href = url; a.download = 'badr-ai-' + Date.now() + '.txt';
  a.click(); URL.revokeObjectURL(url);
}


// ===== FILTRES MULTI-SELECTION =====
var FILTER_TOPICS = [
  "גירושין",
  "מזונות",
  "כתובה",
  "משמורת",
  "חלוקת רכוש",
  "שלום בית"
];
var activeFilters = [];

function renderFilters() {
  var wrap = document.getElementById('filters-pills');
  wrap.innerHTML = '';
  FILTER_TOPICS.forEach(function(topic, i) {
    var pill = document.createElement('div');
    pill.className = 'pill' + (activeFilters.indexOf(topic) >= 0 ? ' on' : '');
    var isOn = activeFilters.indexOf(topic) >= 0;
    pill.innerHTML = (isOn ? '<span class="chk">✔</span>' : '') + topic;
    pill.addEventListener('click', function() {
      var idx = activeFilters.indexOf(topic);
      if (idx >= 0) { activeFilters.splice(idx, 1); }
      else { activeFilters.push(topic); }
      renderFilters();
      updateFilterCount();
    });
    wrap.appendChild(pill);
  });
}

function updateFilterCount() {
  var cnt = document.getElementById('filters-count');
  if (activeFilters.length === 0) {
    cnt.innerHTML = '';
  } else { var qa=document.getElementById("query"); if(qa){qa.placeholder="05E905D005DC05D4 1: 05D405E705DC05D3 05D005EA 05E905D005DC05EA05DA 05D405DE05E905E405D805D905EA 05DB05D005DF...";}
    cnt.innerHTML = activeFilters.length + ' נושאים נבחרו &middot; <button class="btn-clear" onclick="clearFilters()">נקה הכל</button>';
  }
}

function clearFilters() {
  activeFilters = [];
  renderFilters();
  updateFilterCount();
}

renderFilters();
updateFilterCount();

// Bouton "nesse bechinam" -> scroll vers la recherche
var _tb=document.getElementById('try-free-btn');if(_tb)_tb.addEventListener('click', function(e) {
  e.preventDefault();
  document.getElementById('search-zone').scrollIntoView({behavior:'smooth', block:'center'});
  document.getElementById('query').focus();
});


// ===== HISTORIQUE CONVERSATION =====
var chatHistory = []; function resetConv(){var conv=document.getElementById("conversation");if(conv){conv.innerHTML="";conv.style.display="none";}var qa=document.getElementById("query");if(qa){qa.value="";qa.placeholder="שאלה 1: הקלד את שאלתך המשפטית כאן...";}}

function updateChatIndicator() {
  var ind = document.getElementById("chat-indicator");
  var cnt = document.getElementById("chat-count");
  if (chatHistory.length > 0) {
    ind.style.display = "block";
    cnt.textContent = chatHistory.length;
    var qa = document.getElementById("query");
    if(qa){ qa.placeholder = "שאלה " + (chatHistory.length+1) + ": הקלד את שאלתך המשפטית כאן..."; }
  } else {
    var qa=document.getElementById("query"); if(qa){qa.placeholder="שאלה 1: הקלד את שאלתך המשפטית כאן...";}
    ind.style.display = "none";
  }
}

if(document.getElementById("new-chat-btn")) document.getElementById("new-chat-btn").addEventListener("click", function() {
  var qa = document.getElementById("query");
  if(qa){ qa.focus(); }
  window.scrollTo({top: document.body.scrollHeight, behavior: "smooth"});
});

// ===== RESET RECHERCHE =====
function resetSearch() {
  chatHistory = []; resetConv(); updateChatIndicator();
  document.getElementById('query').value = '';
  var qu = document.getElementById('query-upload');
  if (qu) qu.value = '';
  
  
  
  document.getElementById('loading').style.display = 'none';
  var fn = document.getElementById('filename');
  if (fn) fn.textContent = 'לא נבחר קובץ';
  var fi = document.getElementById('file');
  if (fi) fi.value = '';
  selectedFiles=[];renderFiles();
  clearFilters();
}
document.getElementById('reset-search-btn').addEventListener('click', resetSearch);

// Reset complet au clic sur le logo
var logoEl = document.querySelector('.topnav-logo');
if (logoEl) {
  logoEl.addEventListener('click', function(e) {
    e.preventDefault();
    resetSearch();
    window.scrollTo({top:0, behavior:'smooth'});
  });
}

// ===== FAQ ACCORDION =====
document.querySelectorAll('.faq-q').forEach(function(q) {
  q.addEventListener('click', function() {
    q.parentElement.classList.toggle('open');
  });
});

var selectedFiles=[];function showFiles(inp){var lbl=String.fromCharCode(1511,1489,1510,1497,1501);for(var j=0;j<inp.files.length;j++){if(selectedFiles.length<10){selectedFiles.push(inp.files[j]);}else{alert(String.fromCharCode(1504,1497,1514,1503,32,1506,1491,32,49,48,32)+lbl);break;}}inp.value="";renderFiles();}function renderFiles(){var box=document.getElementById("filelist");if(selectedFiles.length===0){box.innerHTML="";return;}var lbl=String.fromCharCode(1511,1489,1510,1497,1501);var h="<div style='color:var(--gold);font-weight:600;margin-bottom:6px;'>"+selectedFiles.length+" "+lbl+":</div>";for(var k=0;k<selectedFiles.length;k++){h+="<div style='background:var(--dark2);padding:6px 10px;border-radius:6px;margin-top:4px;font-size:0.85em;color:var(--text);display:flex;justify-content:space-between;align-items:center;'><span>"+selectedFiles[k].name+"</span><span onclick='removeFile("+k+")' style='cursor:pointer;color:#e74c3c;font-weight:bold;padding:0 8px;'>X</span></div>";}box.innerHTML=h;}function removeFile(i){selectedFiles.splice(i,1);renderFiles();}
function toggleSrc(el){var p=el.querySelector(".src-ext");if(p)p.style.display=p.style.display==="none"?"block":"none";}
function linkSrc(t){return t.replace(/\[\u05de\u05e7\u05d5\u05e8 (\d+)[^\]]*\]/g,function(m,n){return '<a href="#" onclick="jumpSrc('+n+');return false;" style="color:var(--gold);font-weight:700;text-decoration:none;border-bottom:1px dotted var(--gold);">'+m+'</a>';});} function jumpSrc(n){var items=document.querySelectorAll(".source-item");if(items.length>=n){var el=items[n-1];el.scrollIntoView({behavior:"smooth",block:"center"});var ext=el.querySelector(".src-ext");if(ext)ext.style.display="block";el.style.transition="background 0.3s";var orig=el.style.background;el.style.background="var(--gold-dim)";setTimeout(function(){el.style.background=orig;},1500);}}
function scrollToQuery() {
  var qa = document.getElementById('query');
  if(qa){ qa.scrollIntoView({behavior:'smooth'}); qa.focus(); }
}
async function search() {
  var rawQuery = document.getElementById('query').value;
  var query = rawQuery;
  if (activeFilters.length > 0) {
    query = '[' + activeFilters.join(', ') + '] ' + rawQuery;
  }
  var fileList = selectedFiles;
  if (!rawQuery && !fileList.length) { alert('הקלד שאלה או העלה קובץ'); return; }
  document.getElementById('loading').style.display='block';
  
  
  try {
    let resp;
    if (fileList.length) {
      const fd = new FormData();
      for(var fi=0;fi<fileList.length;fi++){fd.append("files",fileList[fi]);}
      fd.append('query', query);
      resp = await fetch('/api/upload', {method:'POST', body:fd});
    } else { var qa=document.getElementById("query"); if(qa){qa.placeholder="05E905D005DC05D4 1: 05D405E705DC05D3 05D005EA 05E905D005DC05EA05DA 05D405DE05E905E405D805D905EA 05DB05D005DF...";}
      const fd = new FormData();
      fd.append('query', query);
      fd.append('history', JSON.stringify(chatHistory));
      resp = await fetch('/api/search', {method:'POST', body:fd});
    }
    const data = await resp.json();
    document.getElementById('loading').style.display='none';
    if (data.error) { alert('שגיאה: '+data.error); return; }
    chatHistory.push({q: rawQuery, a: data.answer}); updateChatIndicator();
    document.getElementById('query').value = '';
    var conv = document.getElementById('conversation'); if(!conv){alert("erreur: div conversation introuvable"); return;}
    conv.style.display='block';
    var turnDiv = document.createElement('div');
    turnDiv.style.cssText='margin-bottom:28px;border-bottom:1px solid var(--border);padding-bottom:20px;';
    var qNum = chatHistory.length;
    var qDiv = '<div style="background:var(--dark2);border-right:3px solid var(--gold);padding:10px 14px;border-radius:8px;margin-bottom:12px;color:var(--text-dim);font-size:0.9em;"><span style="color:var(--gold);font-weight:700;margin-left:8px;">שאלה '+qNum+':</span> ' + rawQuery.replace(/</g,"&lt;") + '</div>';
    var ansBody = linkSrc(data.answer).split(String.fromCharCode(10)).join("<br>").replace(/[*][*]([^*]+)[*][*]/g,"<strong>$1</strong>").replace(/---/g,"<hr>");
    var disc = '<div class="disclaimer">⚠️ המערכת היא כלי עזר בלבד ואינה מהווה ייעוץ משפטי. יש להתייעץ עם עורך דין או טוען רבני מוסמך.</div>';
    var hdr = '<div class="answer-header"><span class="answer-title">⚖️ תשובה משפטית</span><button class="btn-pdf" onclick="exportPDF()">ייצוא PDF</button></div>';
    var draftBtnHtml = '<button class="btn-draft" onclick="generateDraft(this)">צור טיוטה</button>';
    var suiteBtn = '<div style="margin-top:12px;"><button onclick="scrollToQuery()" style="background:linear-gradient(135deg,#1a5c3a,#0f3d26);border:1px solid #2ecc71;color:#2ecc71;padding:10px 16px;border-radius:8px;font-size:0.9em;cursor:pointer;">שאלה בהמשך ↓</button></div>';
    hdr = hdr.replace('</div>', draftBtnHtml + '</div>');
    turnDiv.innerHTML = qDiv + hdr + '<div class="answer-body">' + ansBody + '</div>' + disc + suiteBtn;
    if (data.sources) {
      let html='<div class="sources-title">מקורות:</div>';
      data.sources.forEach(function(s,idx){var ext=s.extrait||'';var num=idx+1;var heb=(ext.match(/[א-ת]/g)||[]).length;var showExt=ext&&heb>20;var driveLink=s.file_id?'https://drive.google.com/file/d/'+s.file_id+'/view':'https://drive.google.com/drive/folders/1wePOCt8HgsSQRw0m1nKPcu6iHMz5lERs?q='+encodeURIComponent(s.filename);html+='<div class="source-item" id="src-item-'+num+'"'+(showExt?' onclick="toggleSrc(this)" style="cursor:pointer;"':'')+'>'+' <span style="display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px;background:var(--gold);color:#0a0e14;border-radius:50%;font-size:0.8em;font-weight:800;margin-left:8px;flex-shrink:0;">'+num+'</span> '+s.filename+' <a href="'+driveLink+'" target="_blank" onclick="event.stopPropagation()" style="color:var(--gold);font-size:0.8em;margin-right:8px;text-decoration:none;border:1px solid var(--gold);padding:2px 6px;border-radius:4px;">פתח ב-Drive ↗</a>'+(showExt?'<div class="src-ext" style="display:none;margin-top:8px;font-size:0.98em;line-height:1.7;color:var(--text);background:rgba(255,213,0,0.12);border-right:3px solid #ffd500;padding:12px;border-radius:6px;direction:rtl;"><div style="background:#ffd500;color:#0a0e14;font-size:0.75em;font-weight:700;padding:2px 8px;border-radius:4px;margin-bottom:8px;display:inline-block;">קטע רלוונטי</div><br>'+ext+'</div>':'')+'</div>';});
      var srcDiv = document.createElement('div');
      srcDiv.innerHTML = html;
      turnDiv.appendChild(srcDiv);
    }
    conv.appendChild(turnDiv);
    window.scrollTo({top:0,behavior:"smooth"});
  } catch(e) {
    document.getElementById('loading').style.display='none';
    alert('שגיאה: '+e.message);
  }
}
</script>
<script>window.addEventListener("load",function(){fetch("/auth/me").then(function(r){return r.json()}).then(function(u){var links=document.querySelector(".topnav-links");if(links){if(u.authenticated&&u.allowed){var a=document.createElement("a");a.href="/mon-historique";a.textContent="היסטוריה";links.appendChild(a);var b=document.createElement("a");b.href="/auth/logout";b.textContent="יציאה";b.style.opacity="0.6";links.appendChild(b);}else{var c=document.createElement("a");c.href="/auth/login";c.textContent="התחבר";links.appendChild(c);}}}).catch(function(){})});</script><script>window.addEventListener("scroll",function(){var h=document.querySelector(".header");if(h){var y=window.scrollY;h.style.backgroundPosition="center "+(y*0.4)+"px";}});</script><script>
function applyTheme(t){
  var b=document.querySelector('.theme-toggle');
  if(t==='light'){document.body.classList.add('light');if(b)b.textContent='🌙';}
  else{document.body.classList.remove('light');if(b)b.textContent='☀️';}
}
function toggleTheme(){
  var next=document.body.classList.contains('light')?'dark':'light';
  localStorage.setItem('toenmor_theme',next);
  applyTheme(next);
}
applyTheme(localStorage.getItem('toenmor_theme')||'dark');
</script>
</body>
</html>'''
@app.post("/api/draft")
async def create_draft(request: Request):
    try:
        data = await request.json()
        answer = data.get("answer", "")
        query = data.get("query", "")
        prompt = f"""אתה עוזר משפטי מומחה. צור טיוטה משפטית מקצועית בעברית על בסיס הניתוח הבא.
שאלת הלקוח: {query}
ניתוח משפטי: {answer[:3000]}
צור טיוטה משפטית מקצועית עם: כותרת, פתיחה, עיקר הטענות, סיכום ובקשה מבית הדין."""
        draft = client_gemini.models.generate_content(model="gemini-2.5-flash", contents=prompt).text
        return JSONResponse({"draft": draft})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/history")
async def history(request: Request):
    user = auth.get_current_user(request)
    if not user or not user.get("allowed"):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return database.get_history(user["email"])


@app.delete("/history/{entry_id}")
async def delete_history(entry_id: int, request: Request):
    user = auth.get_current_user(request)
    if not user or not user.get("allowed"):
        return JSONResponse({"error": "non autorise"}, status_code=401)
    database.delete_entry(user["email"], entry_id)
    return JSONResponse({"ok": True})
@app.get("/mon-historique", response_class=HTMLResponse)
async def mon_historique(request: Request):
    user = auth.get_current_user(request)
    if not user or not user.get("allowed"):
        return RedirectResponse(url="/auth/login")
    return HTMLResponse(open("/home/rag/historique.html").read())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
