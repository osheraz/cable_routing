from setuptools import setup, find_packages

setup(
    name="vlm",  # Replace with your package name
    version="0.1",
    description="test gpt4vision",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/your-repo-url",  # Replace with your repository URL
    packages=find_packages(),  # Automatically find Python packages
    install_requires=[
        line.strip() for line in open("requirements.txt").readlines() if line.strip() and not line.startswith("#")
    ],
    python_requires=">=3.6",  # Specify the Python version requirement
)
