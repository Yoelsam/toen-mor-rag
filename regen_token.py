from google_auth_oauthlib.flow import Flow

SCOPES = ["https://www.googleapis.com/auth/drive"]

flow = Flow.from_client_secrets_file(
    "/home/rag/credentials.json",
    scopes=SCOPES,
    redirect_uri="urn:ietf:wg:oauth:2.0:oob"
)

auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline")
print("\n=== ETAPE 1 : Ouvre ce lien dans ton navigateur ===\n")
print(auth_url)
print("\n=== ETAPE 2 : Autorise, puis colle le code ci-dessous ===\n")

code = input("Code d'autorisation : ").strip()
flow.fetch_token(code=code)

import pickle
with open("/home/rag/token.pickle", "wb") as f:
    pickle.dump(flow.credentials, f)

print("\nTOKEN REGENERE avec scope complet drive !")
print("Scopes:", flow.credentials.scopes)
