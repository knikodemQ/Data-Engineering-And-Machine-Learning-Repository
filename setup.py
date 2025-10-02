from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt', 'r') as f:
        requirements = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('-'):
                package = line.split('==')[0].split('>=')[0].split('<=')[0]
                requirements.append(package)
        return requirements

setup(
    name="data-science-toolkit",
    version="1.0.0",
    author="Data Science Team", 
    author_email="niko@email.com",
    description="Data engineering and machine learning toolkit",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: Beta",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "data-toolkit=src.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["*.py"],
        "src.data": ["*.py"],
        "src.models": ["*.py"], 
        "src.visualization": ["*.py"],
        "src.features": ["*.py"],
    },
)
