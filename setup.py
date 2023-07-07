from setuptools import find_packages, setup

setup(
    name="smfbs",
    version="0.1.1",
    python_requires=">3.8.0",
    author="Dave Fol",
    install_requires=[
        "numpy>=1.24.0",
        "opencv-python>=4.7.0.0",
        "Flask>=2.2.0",
        "waitress>=2.1.0",
        "ndi-python>=5.1.1.0",
        "PyTurboJPEG>=1.7.1"
    ],
    packages=find_packages(),
)
