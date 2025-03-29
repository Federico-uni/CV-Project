import subprocess
import sys

# Costruisci il comando da eseguire
command = [sys.executable, "-m", "pip", "install", "git+https://github.com/parlance/ctcdecode.git"]

# Esegui il comando e attendi che termini
subprocess.check_call(command)
