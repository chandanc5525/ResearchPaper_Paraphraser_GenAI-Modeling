
from setuptools import setup, find_packages

setup(
    name="Research Paper Paraphraser Model",
    version="1.0.0",
    description="Research Paper Writing Tool",
    author="Chandan Chaudhari",
    author_email="chaudhari.chandan22@gmail.com",
    packages=find_packages(),
    package_dir={"": "ResearchPaper-Paraphraser"},
    
    install_requires=["streamlit==1.28.1","nltk==3.8.1","language-tool-python==2.7.1",
                      "transformers==4.30.2","transformers==4.30.2",
                      "sentence-transformers==2.2.2","torch==2.0.1",
                      "huggingface-hub==0.16.4"])