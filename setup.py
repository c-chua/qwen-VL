from setuptools import setup, find_packages

setup(
    name="qwen_vlm",
    version="0.1",
    packages=find_packages(include=["vlm", "retrieval"]),
    py_modules=["vlm_decision_engine"],
)