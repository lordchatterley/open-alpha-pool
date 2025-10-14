# Extra requirements for macOS users

LightGBM requires OpenMP (`libomp`) and a build toolchain (`cmake`, `ninja`).

Install via Homebrew:

```bash
brew install libomp cmake ninja


---

## ⚡ Why this split?

- `requirements.txt` stays **OS-agnostic** (works on Linux/Windows/macOS).  
- macOS users just run one extra step with Homebrew.  
- CI/CD pipelines (Linux runners) don’t choke on `brew install` commands.  

---
