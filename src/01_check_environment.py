# ============ Environment check + manifest ============
# WHY: Records Python, packages, Mac details, and the exact model DIGEST
# (a fingerprint of the model version). If Ollama updates a model later,
# the digest proves which version produced our results.
import json, platform, sys
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

MODELS = ["gpt-oss:20b", "gemma4:26b"]
PACKAGES = ["pandas", "pyarrow", "numpy", "matplotlib", "seaborn",
            "scipy", "statsmodels", "tqdm", "ollama", "huggingface_hub"]

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
OUT = ROOT / "runs" / "environment_manifest.json"

# %% Packages
packages = {}
for p in PACKAGES:
    try:
        packages[p] = version(p)
    except PackageNotFoundError:
        packages[p] = "NOT INSTALLED"

# %% Models
import ollama
try:
    listed = {m.model: m for m in ollama.list().models}
except Exception as e:
    raise SystemExit(f"Cannot reach Ollama. Is the Ollama app open? ({e})")

models = {}
for name in MODELS:
    if name not in listed:
        models[name] = {"status": "NOT PULLED"}
        continue
    m, d = listed[name], listed[name].details
    entry = {"status": "available", "digest": m.digest,
             "size_gb": round(m.size / 1e9, 2),
             "parameter_size": getattr(d, "parameter_size", None),
             "quantization": getattr(d, "quantization_level", None)}
    try:   # load check: reasoning models "think" first, so allow enough tokens
        r = ollama.generate(model=name, prompt="Reply with the single word OK.",
                            options={"temperature": 0, "seed": 42, "num_predict": 1024})
        answer = (r.response or "").strip()
        thinking = (getattr(r, "thinking", None) or "").strip()
        entry["answer"] = answer[:40]
        entry["uses_thinking"] = bool(thinking)
        entry["thinking_chars"] = len(thinking)
        entry["load_check"] = "pass" if answer else "empty response"
    except Exception as e:
        entry["load_check"] = f"FAIL: {e}"
    models[name] = entry

# %% Save + print
manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0], "platform": platform.platform(),
            "machine": platform.machine(), "packages": packages, "models": models}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(manifest, indent=2))

print("Python  :", manifest["python"], "|", manifest["platform"])
for k, v in packages.items():
    print(f"  {k:16s} {v}{'  <-- FIX' if v == 'NOT INSTALLED' else ''}")
for k, v in models.items():
    print(f"\n{k}")
    for kk, vv in v.items():
        print(f"   {kk:15s} {str(vv)[:24]}")
ok = all(v != "NOT INSTALLED" for v in packages.values()) and \
     all(v.get("load_check") == "pass" for v in models.values())
print("\nRESULT:", "ALL GOOD" if ok else "PROBLEM - send screenshot")