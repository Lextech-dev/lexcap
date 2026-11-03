import os
import io
import re
import time
import base64
import secrets
import logging
import asyncio
import numpy as np
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import bcrypt
import httpx
import jwt
from fastapi import FastAPI, HTTPException, Security, Depends, Request, BackgroundTasks
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image
import onnxruntime as rt

from utils.tokenizer_base import Tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s UTC | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE   = (128, 32)
VOCAB      = r"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
MODEL_PATH = "./model/model.onnx"

ORT_INTRA_THREADS = int(os.getenv("ORT_INTRA_THREADS", "1"))
ORT_INTER_THREADS = int(os.getenv("ORT_INTER_THREADS", "1"))

# Supabase + dashboard sessions (set these in Railway → Variables)
SUPABASE_URL         = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
JWT_SECRET           = os.environ.get("JWT_SECRET", "")
JWT_HOURS            = int(os.getenv("JWT_HOURS", "12"))

# How long a validated API key is trusted before re-checking the database
KEY_CACHE_TTL     = int(os.getenv("KEY_CACHE_TTL", "60"))    # valid keys
KEY_CACHE_NEG_TTL = int(os.getenv("KEY_CACHE_NEG_TTL", "30"))  # unknown keys
KEY_CACHE_MAX     = 2000

# Dashboard login brute-force protection
LOGIN_MAX_FAILS = 5
LOGIN_WINDOW    = 900   # seconds

# ── Security config ───────────────────────────────────────────────────────────
BURST_WINDOW = int(os.getenv("BURST_WINDOW", "5"))   # seconds
BURST_MAX    = int(os.getenv("BURST_MAX",    "5"))    # max requests in window

# Known paths — anything else is flagged as scanning
KNOWN_PATHS = {
    '/', '/health', '/solve-captcha-base64',
    '/auth/login', '/me', '/me/regenerate-key', '/favicon.ico',
}

# ── In-memory stores (per worker) ─────────────────────────────────────────────
_ip_activity: dict = {}   # {ip: [timestamps]}
_ip_cache:    dict = {}   # {ip: {geo info}}
_key_cache:   dict = {}   # {api_key: (user_dict | None, fetched_at)}
_login_fails: dict = {}   # {ip: [timestamps]}
request_count = 0

db: httpx.AsyncClient | None = None   # Supabase REST client, created at startup

# Compared against when an email doesn't exist, so response time doesn't reveal which emails are real
_DUMMY_HASH = bcrypt.hashpw(b"dummy-password", bcrypt.gensalt(rounds=12))


def client_ip(request: Request) -> str:
    """Real client IP — first entry of X-Forwarded-For behind Railway's proxy."""
    fwd = request.headers.get("X-Forwarded-For")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _parse_ts(value):
    """Parse a Supabase timestamptz string tolerantly (variable fractional digits)."""
    if not value:
        return None
    s = value.replace("Z", "+00:00")
    m = re.match(r"^(.*?\.)(\d+)(.*)$", s)
    if m:
        s = m.group(1) + m.group(2).ljust(6, "0")[:6] + m.group(3)
    return datetime.fromisoformat(s)


# ── Database helpers ──────────────────────────────────────────────────────────
USER_FIELDS = "id,name,email,api_key,plan,active,expires_at,solves_total,created_at"

async def fetch_user(**eq) -> dict | None:
    """Fetch one user row by exact match, e.g. fetch_user(api_key='lxt-...')."""
    params = {"select": USER_FIELDS, "limit": "1"}
    for col, val in eq.items():
        params[col] = f"eq.{val}"
    r = await db.get("/rest/v1/users", params=params)
    r.raise_for_status()
    rows = r.json()
    return rows[0] if rows else None


async def lookup_api_key(api_key: str) -> dict | None:
    """
    Resolve an API key to a user. Cached per worker so the database is not
    hit on every solve. Falls back to a stale cached user if Supabase is down.
    """
    now    = time.time()
    cached = _key_cache.get(api_key)
    if cached:
        user, fetched = cached
        ttl = KEY_CACHE_TTL if user else KEY_CACHE_NEG_TTL
        if now - fetched < ttl:
            return user

    try:
        user = await fetch_user(api_key=api_key)
    except Exception as e:
        logger.error(f"❌ Supabase lookup failed: {e}")
        if cached and cached[0]:
            return cached[0]           # serve stale rather than fail the solve
        raise HTTPException(status_code=503, detail="Auth service unavailable")

    if len(_key_cache) >= KEY_CACHE_MAX:
        _key_cache.clear()
    _key_cache[api_key] = (user, now)
    return user


def drop_cached_keys_for(user_id: str):
    for k in [k for k, (u, _) in _key_cache.items() if u and u["id"] == user_id]:
        _key_cache.pop(k, None)


async def log_usage(user_id: str, ip: str, success: bool, ms: int):
    """Runs after the response is sent — never slows down a solve."""
    try:
        calls = [db.post("/rest/v1/usage_log", json={
            "user_id": user_id, "ip": ip, "success": success, "processing_ms": ms
        })]
        if success:
            calls.append(db.post("/rest/v1/rpc/record_solve", json={
                "p_user_id": user_id, "p_ip": ip
            }))
        await asyncio.gather(*calls)
    except Exception as e:
        logger.error(f"❌ Usage log failed: {e}")


async def count_solves_today(user_id: str) -> int:
    start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    r = await db.get(
        "/rest/v1/usage_log",
        params={"user_id": f"eq.{user_id}", "success": "eq.true",
                "created_at": f"gte.{start.isoformat()}", "select": "id"},
        headers={"Prefer": "count=exact", "Range": "0-0"},
    )
    try:
        return int(r.headers.get("content-range", "*/0").split("/")[-1])
    except ValueError:
        return 0


# ── API key auth (used by the extension) ──────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_key(key: str = Security(api_key_header)) -> dict:
    if not key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    user = await lookup_api_key(key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    if not user["active"]:
        raise HTTPException(status_code=403, detail="Account disabled")

    expires = _parse_ts(user.get("expires_at"))
    if expires and expires < datetime.now(timezone.utc):
        raise HTTPException(status_code=403, detail="Subscription expired")

    return user


# ── Dashboard auth (JWT) ──────────────────────────────────────────────────────
def make_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


async def current_user(request: Request) -> dict:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not logged in")
    try:
        payload = jwt.decode(auth[7:], JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Session expired — log in again")

    try:
        user = await fetch_user(id=payload["sub"])
    except Exception as e:
        logger.error(f"❌ Session lookup failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")
    if not user or not user["active"]:
        raise HTTPException(status_code=401, detail="Account unavailable")
    return user


# ── IP geolocation — cached per IP ───────────────────────────────────────────
async def get_ip_info(ip: str) -> dict:
    """
    Lookup IP info via ip-api.com (free, no key needed).
    Result cached so only first request per IP hits the external API.
    Returns empty dict on failure — never blocks a request.
    """
    if ip in _ip_cache:
        return _ip_cache[ip]
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            res = await client.get(
                f"http://ip-api.com/json/{ip}",
                params={"fields": "country,regionName,city,isp,org,proxy,hosting"}
            )
            info = res.json() if res.status_code == 200 else {}
    except Exception:
        info = {}
    _ip_cache[ip] = info
    return info

# ── Load model ────────────────────────────────────────────────────────────────
logger.info("Loading ONNX model...")
tokenizer = Tokenizer(VOCAB)

opts = rt.SessionOptions()
opts.intra_op_num_threads     = ORT_INTRA_THREADS
opts.inter_op_num_threads     = ORT_INTER_THREADS
opts.execution_mode           = rt.ExecutionMode.ORT_SEQUENTIAL
opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.enable_mem_pattern       = True
opts.enable_cpu_mem_arena     = True

session = rt.InferenceSession(
    MODEL_PATH,
    sess_options=opts,
    providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
logger.info("✅ ONNX model ready")

# ── Inference ─────────────────────────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    x = np.ascontiguousarray(image, dtype=np.float32)
    x = (x / 255.0 - 0.5) / 0.5
    return x.transpose(2, 0, 1)[np.newaxis, :]


def solve_image(image: Image.Image) -> str:
    x      = preprocess(image)
    logits = session.run(None, {input_name: x})[0]
    e      = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs  = e / e.sum(axis=-1, keepdims=True)
    preds, _ = tokenizer.decode(probs)
    return preds[0]


# ── Warmup ────────────────────────────────────────────────────────────────────
def warmup():
    logger.info("Warming up model...")
    dummy = Image.new("RGB", IMG_SIZE, color=(128, 128, 128))
    for _ in range(3):
        solve_image(dummy)
    logger.info("✅ Warmup complete — model is hot")


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global db
    missing = [n for n, v in (
        ("SUPABASE_URL", SUPABASE_URL),
        ("SUPABASE_SERVICE_KEY", SUPABASE_SERVICE_KEY),
        ("JWT_SECRET", JWT_SECRET),
    ) if not v]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    db = httpx.AsyncClient(
        base_url=SUPABASE_URL,
        headers={
            "apikey":        SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type":  "application/json",
        },
        timeout=5.0,
    )
    warmup()
    yield
    await db.aclose()


# ── Schemas ───────────────────────────────────────────────────────────────────
class SolveRequest(BaseModel):
    image_base64: str

class SolveResponse(BaseModel):
    success: bool
    text: str = ""
    processing_time: float = 0.0
    error: str = ""

class LoginRequest(BaseModel):
    email: str
    password: str


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CAPTCHA Solver API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Access + security monitoring middleware ───────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    global request_count
    request_count += 1

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    ip        = client_ip(request)
    api_key   = request.headers.get("X-API-Key", "none")
    key_hint  = api_key[:8] + "…" if len(api_key) > 8 else api_key
    req_num   = request_count
    path      = request.url.path

    # ── Unknown path — endpoint scanning ─────────────────────────────────────
    if path not in KNOWN_PATHS:
        logger.warning(
            f"⚠️  UNKNOWN PATH | IP={ip} | {request.method} {path}"
        )

    # ── Burst detection ───────────────────────────────────────────────────────
    now = time.time()
    if ip not in _ip_activity:
        _ip_activity[ip] = []
    _ip_activity[ip] = [t for t in _ip_activity[ip] if now - t < BURST_WINDOW]
    _ip_activity[ip].append(now)

    if len(_ip_activity[ip]) > BURST_MAX:
        logger.warning(
            f"🚨 BURST DETECTED | IP={ip} | "
            f"{len(_ip_activity[ip])} requests in {BURST_WINDOW}s"
        )

    # ── IP geolocation — cached, never blocks the request ────────────────────
    info       = await get_ip_info(ip)
    country    = info.get("country",  "unknown")
    isp        = info.get("isp",      "unknown")
    is_proxy   = info.get("proxy",    False)
    is_hosting = info.get("hosting",  False)

    if is_proxy or is_hosting:
        logger.warning(
            f"🚨 SUSPICIOUS IP | IP={ip} | country={country} | "
            f"isp={isp} | proxy={is_proxy} | datacenter={is_hosting}"
        )

    # ── Process request ───────────────────────────────────────────────────────
    start    = time.time()
    response = await call_next(request)
    elapsed  = time.time() - start

    logger.info(
        f"[#{req_num}] {timestamp} UTC | IP={ip} | country={country} | "
        f"isp={isp} | key={key_hint} | {request.method} {path} | {elapsed:.3f}s"
    )

    return response


# ── Frontend ──────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def index():
    return FileResponse("index.html")


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":          "ok",
        "device":          "cpu",
        "model":           "ONNX INT8",
        "quantized":       True,
        "workers":         os.getenv("WEB_CONCURRENCY", "1"),
        "intra_threads":   ORT_INTRA_THREADS,
        "requests_served": request_count,
    }


# ── Dashboard auth endpoints ──────────────────────────────────────────────────
@app.post("/auth/login")
async def login(req: LoginRequest, request: Request):
    ip  = client_ip(request)
    now = time.time()

    fails = [t for t in _login_fails.get(ip, []) if now - t < LOGIN_WINDOW]
    _login_fails[ip] = fails
    if len(fails) >= LOGIN_MAX_FAILS:
        logger.warning(f"🚨 LOGIN LOCKOUT | IP={ip} | {len(fails)} failed attempts")
        raise HTTPException(status_code=429, detail="Too many attempts — try again later")

    email = req.email.strip().lower()
    try:
        r = await db.get("/rest/v1/users", params={
            "email": f"eq.{email}", "select": "id,password_hash,active", "limit": "1"
        })
        r.raise_for_status()
        rows = r.json()
    except Exception as e:
        logger.error(f"❌ Login lookup failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

    row = rows[0] if rows else None
    stored = row["password_hash"].encode() if row else _DUMMY_HASH

    def _check() -> bool:
        try:
            return bcrypt.checkpw(req.password.encode(), stored)
        except ValueError:          # e.g. password longer than bcrypt's 72-byte limit
            return False

    # bcrypt is CPU-heavy — keep it off the event loop
    ok = await asyncio.to_thread(_check) and row is not None

    if not ok or not row["active"]:
        _login_fails[ip].append(now)
        logger.warning(f"⚠️  LOGIN FAILED | IP={ip} | email={email}")
        raise HTTPException(status_code=401, detail="Invalid email or password")

    _login_fails.pop(ip, None)
    logger.info(f"✅ LOGIN | IP={ip} | email={email}")
    return {"token": make_token(row["id"])}


@app.get("/me")
async def me(user: dict = Depends(current_user)):
    return {
        "name":         user["name"],
        "email":        user["email"],
        "plan":         user["plan"],
        "api_key":      user["api_key"],
        "active":       user["active"],
        "expires_at":   user["expires_at"],
        "solves_total": user["solves_total"],
        "solves_today": await count_solves_today(user["id"]),
        "created_at":   user["created_at"],
    }


@app.post("/me/regenerate-key")
async def regenerate_key(user: dict = Depends(current_user)):
    new_key = "lxt-" + secrets.token_hex(16)
    r = await db.patch("/rest/v1/users", params={"id": f"eq.{user['id']}"},
                       json={"api_key": new_key})
    r.raise_for_status()
    drop_cached_keys_for(user["id"])     # this worker; others expire within the TTL
    logger.info(f"🔑 KEY REGENERATED | email={user['email']}")
    return {"api_key": new_key}


# ── Solver endpoint ───────────────────────────────────────────────────────────
@app.post("/solve-captcha-base64", response_model=SolveResponse)
async def solve(
    req: SolveRequest,
    request: Request,
    background: BackgroundTasks,
    user: dict = Depends(verify_key),
):
    start = time.time()
    ip    = client_ip(request)
    try:
        raw = req.image_base64
        if "," in raw:
            raw = raw.split(",", 1)[1]

        image = Image.open(io.BytesIO(base64.b64decode(raw)))
        # Inference runs in a thread so it never blocks the event loop
        text  = (await asyncio.to_thread(solve_image, image)).strip()[:5]

        # Small delay before returning — avoids triggering rate limits
        # on the target site from responses arriving too fast
        await asyncio.sleep(0.5)

        elapsed = time.time() - start
        logger.info(f"✅ Solved: '{text}' in {elapsed:.3f}s | user={user['email']}")
        background.add_task(log_usage, user["id"], ip, True, int(elapsed * 1000))

        return SolveResponse(success=True, text=text, processing_time=elapsed)

    except Exception as e:
        logger.error(f"❌ Error: {e} | user={user['email']}")
        background.add_task(log_usage, user["id"], ip, False, int((time.time() - start) * 1000))
        return SolveResponse(
            success=False,
            error=str(e),
            processing_time=time.time() - start
        )
