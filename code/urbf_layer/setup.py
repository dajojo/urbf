from setuptools import setup, find_packages

setup(
    name="urbf_layer",
    version='0.0.1',
    author='Daniel Jost',
    packages=find_packages(),
    install_requires=['torch>=2.0','numpy'],
)