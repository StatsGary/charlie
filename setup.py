from setuptools import setup, find_packages

setup(
    name="ensemliPy",
    version="0.0.1",
    author="Gary Hutson",
    author_email="hutsons-hacks@outlook.com",
    description="A package for ensemble modeling methods",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/your-repo",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.0",        # For DataFrame operations
        "numpy>=1.20",        # For numerical computations
        "scikit-learn>=0.24"  # For machine learning utilities
    ],
    extras_require={
        "dev": ["pytest>=7.0"]  # Development and testing dependencies
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)

