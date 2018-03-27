#!/usr/bin/env python3

import os
from setuptools import setup


def read(fname):
  return open(os.path.join(os.path.dirname(__file__), fname)).read()


a = setup(
  name = "glove",
  version = "0.0.1",
  description = "Vector representations for words extracted from tweets.",
  license = "MIT License",
  long_description=read('README.md'),
  keywords = "glove embeddings word vectors stanford",
  url = "https://github.com/DopplerFoundation/example-app-glove",
  packages = ['app'],
  install_requires = [
    "gensim"
  ],
  classifiers = [
    "Topic :: Utilities",
    "License :: MIT License"
  ]
)