---
name: block-ssh-key-edits
enabled: true
event: file
action: block
conditions:
  - field: file_path
    operator: regex_match
    pattern: (id_rsa|id_ed25519|id_ecdsa|id_dsa|\.pem|\.key)$
---

**Sensitive file edit blocked: SSH key or certificate**

This file appears to be an SSH key or TLS certificate/key.

**These files should never be edited by automation.**
- SSH keys should be generated with ssh-keygen
- Certificates should be managed by your CA (Step CA at ca.home:8433)
