import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "social_int_analyses",
    version = "0.0.1",
    author = "ESay",
    author_email = "ellaksay@gmail.com",
    description = ("Combine two-photon with tracking data"),
    license = "BSD",
    keywords = "",
    url = "https://github.com/GiocomoLab/TwoPUtils",
    packages=find_packages(), #['social_int_analyses'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)