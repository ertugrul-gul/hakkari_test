import platform
import json
import os

vscode_dir = ".vscode"
settings_path = os.path.join(vscode_dir, "settings.json")

if not os.path.exists(vscode_dir):
    os.makedirs(vscode_dir)

python_path = "./.venv/Lib/site-packages" if platform.system() == "Windows" else "./.venv/lib/pythonX.X/site-packages"

settings = {
    "python.analysis.extraPaths": [python_path]
}

with open(settings_path, "w") as f:
    json.dump(settings, f, indent=4)

print(f"Settings saved to {settings_path}")
