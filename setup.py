from setuptools import find_packages, setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name='ELFit',
    version='v0.2',
    description='Echelle Line Fitting GUI',
    python_requires='==3.8.*',
    long_description=readme(),
    platforms=['any'],
    packages=['ELFit'],
    url='https://github.com/kennethcarrell/ELFit',
    license='GPL-3.0',
    author='Kenneth Carrell',
    author_email='kennethcarrell@gmail.com',
    install_requires=['numpy>=1.19', 'astropy>=4.2', 'matplotlib>=3.3', 'specutils>=1.2'],
    keywords=['astronomy', 'absorption line fitting', 'echelle spectra', 'stars'],
    classifiers=['Intended Audience :: Science/Research', 'Topic :: Scientific/Engineering :: Astronomy', 'Development Status :: 5 - Production/Stable', 'Operating System :: OS Independent', 'Programming Language :: Python'],
)
