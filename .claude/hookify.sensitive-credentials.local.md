---
name: block-credentials-edits
enabled: true
event: file
action: block
conditions:
  - field: file_path
    operator: regex_match
    pattern: (credentials|secrets?|tokens?)\.(json|ya?ml|toml|ini|conf)$
---

**Sensitive file edit blocked: credentials/secrets file**

This file likely contains authentication credentials or secret values.

**Before proceeding:**
- Confirm this edit is safe and intentional
- Never hardcode secrets — use environment variables or a secret manager
- Ensure this file is in .gitignore
