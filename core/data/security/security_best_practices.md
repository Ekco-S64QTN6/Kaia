#### D. Security Best Practices (for `data/security_best_practices.md`)


```markdown
## SSH Hardening
1. Disable root login:
```bash
PermitRootLogin no

Use key-based auth:

Bash

PasswordAuthentication no

Firewall Basics

Bash

sudo ufw allow 22/tcp  # SSH
sudo ufw enable
