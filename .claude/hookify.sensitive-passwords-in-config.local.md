---
name: block-passwords-in-files
enabled: true
event: file
action: block
conditions:
  - field: new_text
    operator: regex_match
    pattern: (password|secret|token|api_key|apikey|API_KEY|SECRET_KEY|PRIVATE_KEY)\s*[:=]\s*["\']?[^\s"\']{8,}
---

**Potential secret detected in file edit**

A value that looks like a password, secret, token, or API key is being written to a file.

**Best practices:**
- Use environment variables instead of hardcoding secrets
- Reference secrets from a vault or .env file (which should be in .gitignore)
- Never commit secrets to version control
