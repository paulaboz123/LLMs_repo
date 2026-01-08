# One-command deployment (Windows PowerShell):
#   powershell -ExecutionPolicy Bypass -File .\infra\deploy.ps1

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r infra\requirements.deploy.txt
python infra\deploy.py
