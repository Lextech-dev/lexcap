"""
create_user.py — manage LexTechSolver accounts from your own machine.

Setup (once):
    pip install bcrypt httpx
    export SUPABASE_URL="https://xxxx.supabase.co"
    export SUPABASE_SERVICE_KEY="your-service_role-key"

Usage:
    python create_user.py create  --name "Jane Doe" --email jane@example.com --days 30
    python create_user.py list
    python create_user.py extend  --email jane@example.com --days 30    # after an M-Pesa payment
    python create_user.py disable --email jane@example.com              # blocks within ~60s
    python create_user.py enable  --email jane@example.com
"""
import argparse
import getpass
import os
import re
import secrets
import sys
from datetime import datetime, timedelta, timezone

import bcrypt
import httpx

URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
if not URL or not KEY:
    sys.exit("Set SUPABASE_URL and SUPABASE_SERVICE_KEY first.")

db = httpx.Client(
    base_url=URL,
    headers={"apikey": KEY, "Authorization": f"Bearer {KEY}",
             "Content-Type": "application/json", "Prefer": "return=representation"},
    timeout=10.0,
)


def parse_ts(value):
    if not value:
        return None
    s = value.replace("Z", "+00:00")
    m = re.match(r"^(.*?\.)(\d+)(.*)$", s)
    if m:
        s = m.group(1) + m.group(2).ljust(6, "0")[:6] + m.group(3)
    return datetime.fromisoformat(s)


def get_user(email):
    r = db.get("/rest/v1/users", params={"email": f"eq.{email.lower()}", "select": "*"})
    r.raise_for_status()
    rows = r.json()
    if not rows:
        sys.exit(f"No user with email {email}")
    return rows[0]


def cmd_create(a):
    password = a.password or getpass.getpass("Password for the user: ")
    if len(password) < 8:
        sys.exit("Password must be at least 8 characters.")
    if len(password.encode()) > 72:
        sys.exit("Password must be 72 bytes or fewer (bcrypt limit).")
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    body = {"name": a.name, "email": a.email.lower(), "password_hash": hashed}
    if a.days:
        body["expires_at"] = (datetime.now(timezone.utc) + timedelta(days=a.days)).isoformat()
    r = db.post("/rest/v1/users", json=body)
    if r.status_code >= 400:
        sys.exit(f"Failed: {r.text}")
    u = r.json()[0]
    print(f"\n✅ Created {u['email']}")
    print(f"   API key : {u['api_key']}")
    print(f"   Expires : {u['expires_at'] or 'never'}")
    print("   Send the user their email, password and API key.")


def cmd_list(_):
    r = db.get("/rest/v1/users", params={
        "select": "name,email,active,expires_at,solves_total,last_solve_at,last_ip",
        "order": "created_at.asc"})
    r.raise_for_status()
    for u in r.json():
        state = "active" if u["active"] else "DISABLED"
        print(f"{u['email']:<32} {state:<9} solves={u['solves_total']:<7} "
              f"expires={u['expires_at'] or 'never'}  last_ip={u['last_ip'] or '-'}")


def cmd_extend(a):
    u = get_user(a.email)
    base = max(datetime.now(timezone.utc), parse_ts(u["expires_at"]) or datetime.now(timezone.utc))
    new = (base + timedelta(days=a.days)).isoformat()
    r = db.patch("/rest/v1/users", params={"email": f"eq.{a.email.lower()}"},
                 json={"expires_at": new, "active": True})
    r.raise_for_status()
    print(f"✅ {a.email} now expires {new}")


def set_active(email, value):
    r = db.patch("/rest/v1/users", params={"email": f"eq.{email.lower()}"},
                 json={"active": value})
    r.raise_for_status()
    print(f"✅ {email} {'enabled' if value else 'disabled'}")


p = argparse.ArgumentParser(description="Manage LexTechSolver users")
sub = p.add_subparsers(dest="cmd", required=True)

c = sub.add_parser("create")
c.add_argument("--name", required=True)
c.add_argument("--email", required=True)
c.add_argument("--password")
c.add_argument("--days", type=int, help="subscription length in days (omit for no expiry)")
c.set_defaults(fn=cmd_create)

sub.add_parser("list").set_defaults(fn=cmd_list)

e = sub.add_parser("extend")
e.add_argument("--email", required=True)
e.add_argument("--days", type=int, default=30)
e.set_defaults(fn=cmd_extend)

for name, val in (("disable", False), ("enable", True)):
    s = sub.add_parser(name)
    s.add_argument("--email", required=True)
    s.set_defaults(fn=lambda a, v=val: set_active(a.email, v))

args = p.parse_args()
args.fn(args)
