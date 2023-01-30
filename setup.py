import setuptools
import sys
from setuptools import setup
from warnings import warn
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='ADlasso',
    version='0.0.1', #Jan-25-2023
    packages=['ADlasso',],
    license='MIT',
    author='yincheng_chen',
    author_email = 'yin.cheng.23@gmail.com',
    url = 'https://github.com/YinchengChen23/ADlasso',
    setup_requires = ['numpy>=1.17','torch'],
    install_requires=['numpy>=1.17','torch','pandas>=0.25','scipy','scikit-learn>=1.2', 'matplotlib','seaborn'],
    long_description=long_description,
    long_description_content_type='text/markdown'
)