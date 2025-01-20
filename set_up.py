from setuptools import setup, find_packages

setup(
    name="arxiv-harvester",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "backoff>=2.2.1",
    ],
    python_requires=">=3.7",
)