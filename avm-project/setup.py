from setuptools import setup, find_packages
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQS_PATH = ROOT / "requirements.txt"

def load_requirements(path: Path):
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="avm_project",
    version="0.0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=load_requirements(REQS_PATH),
)
