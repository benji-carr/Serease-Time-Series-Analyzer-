from setuptools import setup, find_packages
from pathlib import Path

HERE = Path(__file__).parent

def read_requirements():
    req_path = HERE / "requirements.txt"
    if not req_path.exists():
        return []
    lines = req_path.read_text(encoding="utf-8").splitlines()
    reqs = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # ignore editable installs / constraints / local paths
        if line.startswith("-"):
            continue
        reqs.append(line)
    return reqs

setup(
    name="Serease",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=read_requirements(),
    python_requires=">=3.10",
)
