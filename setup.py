from setuptools import setup, find_packages

setup(
    name='wordvec_app',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'nltk',
        'tqdm',
        'gensim'
    ],
)
