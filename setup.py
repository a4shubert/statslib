import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="statslib",
    version="0.0.1",
    author="Alexander Shubert",
    author_email="ashubertt@gmail.com",
    description="Python library for rapid statistical and ML modelling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="localhost//statslib",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
)
