#!/usr/bin/env python3

import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


a = setup(
    name="similar-words",
    version="0.0.1",
    description="Find similar words.",
    license="Apache License",
    long_description=read('README.md'),
    keywords="similarity similar words nlp",
    url="https://github.com/DopplerFoundation/example-app-similar-words",
    packages=['app'],
    install_requires=[
        "gensim"
    ],
    classifiers=[
        "Topic :: Utilities",
        "License :: Apache License"
    ]
)
