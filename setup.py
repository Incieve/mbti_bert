#!/usr/bin/env python
from distutils.core import setup
from setuptools import setup, find_packages

setup(name='bert_mbti',
      version='1.0',
      description='',
      author='',
      author_email='',
      packages=find_packages(),
      install_requires=['torch', 'numpy',"pandas",
                        'matplotlib', 'seaborn', 'transformers==3.0.0',
                        'scikit-learn'],
      entry_points='''
            [console_scripts]
            train_bert=pre_bert.model.main
            use_bert=pre_bert.model.use_bert
      ''',
)