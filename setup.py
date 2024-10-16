from setuptools import setup, find_packages

setup(
    name="Dummy",
    version="0.1.0",
    author="Your Name",
    author_email="pathompong.workspace@gmail.com",
    description="A Python library to generate dummy data for Polars or Pandas DataFrames",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/p4zaa/Dummy",
    install_requires=[
        "polars",
        "pandas"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)