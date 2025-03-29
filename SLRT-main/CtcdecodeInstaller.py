import subprocess
import sys

command = [
    sys.executable,
    "-m", "pip",
    "install",
    "--no-build-isolation",
    "git+https://github.com/parlance/ctcdecode.git"
]

subprocess.check_call(command)
