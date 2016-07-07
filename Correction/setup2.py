from setuptools import setup
import sys

sys.setrecursionlimit(50000)


setup(
    app=["CorrectionWin.py"],
	setup_requires = ["py2app"],
)