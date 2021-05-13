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
    include_package_data=True,
	install_requires=[
	'matplotlib==3.3.4',
	'pandas==1.1.5',
	'seaborn==0.11.1',
	'statsmodels==0.12.2',
	'tensorflow==2.4.1',
	'openpyxl==3.0.7',
	'scikit-learn==0.24.2',
	'beautifulsoup4==4.9.3',
]
)
