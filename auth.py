import os

from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse

REDIRECT_URI = "https://toen-mor.com/auth/callback"

ALLOWED_EMAILS = {
    e.strip().lower()
    for e in os.environ.get("ALLOWED_EMAILS", "").split(",")
    if e.strip()
}

oauth = OAuth()
oauth.register(
    name="google",
    client_id=os.environ["GOOGLE_CLIENT_ID"],
    client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

router = APIRouter()


def _is_allowed(email):
    return True if not ALLOWED_EMAILS else email.lower() in ALLOWED_EMAILS


@router.get("/auth/login")
async def login(request: Request):
    return await oauth.google.authorize_redirect(request, REDIRECT_URI)


@router.get("/auth/callback")
async def callback(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception:
        return RedirectResponse(url="/?error=auth")

    userinfo = token.get("userinfo")
    if not userinfo or not userinfo.get("email"):
        return RedirectResponse(url="/?error=auth")

    email = userinfo["email"].lower()
    request.session["user"] = {
        "email": email,
        "name": userinfo.get("name", ""),
        "picture": userinfo.get("picture", ""),
        "allowed": _is_allowed(email),
    }
    return RedirectResponse(url="/")


@router.get("/auth/logout")
async def logout(request: Request):
    request.session.pop("user", None)
    return RedirectResponse(url="/")


@router.get("/auth/me")
async def me(request: Request):
    user = request.session.get("user")
    if not user:
        return JSONResponse({"authenticated": False})
    return JSONResponse({"authenticated": True, **user})


def get_current_user(request: Request):
    return request.session.get("user")
