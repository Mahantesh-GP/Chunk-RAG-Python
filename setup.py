from setuptools import setup, find_packages

setup(
    name="rag-eval",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "llama-index==0.14.12",
        "openai>=0.27.0",
        "azure-search-documents>=11.4.0",
        "python-dotenv",
        "tqdm",
        "pytest",
    ],
    python_requires=">=3.8",
)
