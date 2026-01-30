from setuptools import setup, find_packages

setup(
    name="Serease",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "statsmodels",
        "scipy",
        "chardet",
    ],
)
