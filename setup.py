from setuptools import setup, find_packages

setup(
    name="lrux",
    version="0.0.1",
    description="Fast low-rank updates (LRU) of matrix determinants and pfaffians in JAX",
    author="Ao Chen, Christopher Roth",
    author_email="chenao.phys@gmail.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=["jax>=0.4.4"],
)
