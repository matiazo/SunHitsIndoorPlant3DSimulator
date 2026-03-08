#!/usr/bin/env python3
"""Create a long-lived access token for Home Assistant."""
import json
import secrets
import hashlib
from datetime import datetime, timezone

AUTH_FILE = "/config/.storage/auth"

# Load current auth data
with open(AUTH_FILE) as f:
    auth_data = json.load(f)

# Find the user (first user)
users = auth_data.get("data", {}).get("users", [])
if not users:
    print("ERROR: No users found")
    exit(1)

user_id = users[0]["id"]

# Generate a new token
raw_token = secrets.token_hex(64)
token_hash = hashlib.sha512(raw_token.encode()).hexdigest()

# Create new refresh token entry
new_token = {
    "id": secrets.token_hex(16),
    "user_id": user_id,
    "client_id": None,
    "client_name": "sun-plant-monitor",
    "client_icon": None,
    "token_type": "long_lived_access_token",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "access_token_expiration": 315360000.0,
    "token": token_hash,
    "jwt_key": secrets.token_hex(64),
    "last_used_at": datetime.now(timezone.utc).isoformat(),
    "last_used_ip": None,
    "expire_at": None,
    "credential_id": None,
    "version": "2025.9.3"
}

# Add to refresh tokens
auth_data["data"]["refresh_tokens"].append(new_token)

# Save
with open(AUTH_FILE, "w") as f:
    json.dump(auth_data, f, indent=2)

print(raw_token)
