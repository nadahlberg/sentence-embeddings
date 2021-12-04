from setuptools import setup

setup(
    name='sentence-embeddings',
    version='0.0.1',    
    description='Search and visualize sentence embeddings.',
    url='https://github.com/nadahlberg/sentence-embeddings',
    author='Nathan Dahlberg',
    author_email='nadahlberg@gmail.com',
    package_dir={'': 'src'},
    packages=['sentence_embeddings'],
    install_requires=[
        'faiss-cpu',
        'sklearn',
        'plotly',
    ],
)