from setuptools import setup, find_packages


setup(
    name='torchninja',
    version='0.1.0',
    author='Silvan Ferreira',
    author_email='silvanfj@gmail.com',
    description='Tools for training PyTorch models',
    url='https://github.com/silvaan/torchninja',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    install_requires=[
        'matplotlib',
        'numpy',
        'torch',
        'torchvision',
        'tensorboard',
        'tqdm',
    ],
)
