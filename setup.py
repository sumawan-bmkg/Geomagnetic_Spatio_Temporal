"""
Setup script for Spatio-Temporal Earthquake Precursor Analysis Project
"""
from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements
def read_requirements():
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='spatio-precursor-analysis',
    version='1.0.0',
    description='Spatio-Temporal Earthquake Precursor Analysis using Geomagnetic Data',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    author='Spatio Precursor Project Team',
    author_email='your.email@example.com',
    url='https://github.com/your-username/spatio-precursor-project',
    
    # Package configuration
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Optional dependencies
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.12.0',
            'black>=21.0.0',
            'flake8>=3.9.0',
        ],
        'plotting': [
            'seaborn>=0.11.0',
        ],
        'parallel': [
            'joblib>=1.1.0',
        ],
        'progress': [
            'tqdm>=4.62.0',
        ]
    },
    
    # Python version requirement
    python_requires='>=3.7',
    
    # Entry points for command-line scripts
    entry_points={
        'console_scripts': [
            'spatio-test=test_installation:run_all_tests',
        ],
    },
    
    # Package data
    package_data={
        'preprocessing': ['*.yaml'],
    },
    
    # Additional files to include
    include_package_data=True,
    
    # Classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    
    # Keywords
    keywords='earthquake precursor geomagnetic ULF scalogram CWT seismology',
    
    # Project URLs
    project_urls={
        'Bug Reports': 'https://github.com/your-username/spatio-precursor-project/issues',
        'Source': 'https://github.com/your-username/spatio-precursor-project',
        'Documentation': 'https://github.com/your-username/spatio-precursor-project/wiki',
    },
)