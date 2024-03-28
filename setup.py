from setuptools import setup, find_packages

import os
os.environ["SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL"] = "True"


with open("README.md", "r", encoding='utf-8') as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding='utf-8') as f:
    requirements = f.read().strip().split("\n")


setup(
    name="easse",
    version="0.3",
    description="Easier Automatic Sentence Simplification Evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3.7",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    author="Fernando Alva-Manchego <feralvam@gmail.com>, Louis Martin <louismartincs@gmail.com>, Regina Stodden <regina.stodden@hhu.de>",
    url="https://github.com/feralvam/easse",
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    test_suite="tests",
    entry_points={"console_scripts": ["easse = easse.cli:cli"]},
    install_requires=requirements,
)
