from setuptools import setup, find_packages

setup(
    name="function_regression",
    version='0.0.2',
    author='Daniel Jost',
    packages=find_packages(),
    install_requires=['torch>=2.0','numpy','pmlb','experiment-utilities','scikit-learn','matplotlib','ucimlrepo','imageio'],
)