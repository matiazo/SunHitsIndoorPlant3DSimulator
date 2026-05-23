---
name: block-env-file-edits
enabled: true
event: file
action: block
conditions:
  - field: file_path
    operator: regex_match
    pattern: \.env(\.\w+)?$
---

**Sensitive file edit blocked: .env file**

This file may contain secrets, API keys, or environment-specific configuration.

**Before proceeding:**
- Confirm this edit is intentional
- Ensure secrets are not being hardcoded
- Verify this file is listed in .gitignore
