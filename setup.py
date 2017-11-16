from Cython.Build import cythonize
import numpy
import os.path

# immediately below is stupid hackery for setuptools to work with Cython
import distutils.extension
from distutils.extension import Extension as _Extension
from setuptools import setup
distutils.extension.Extension = _Extension
distutils.command.build_ext.Extension = _Extension
Extension = _Extension
from Cython.Distutils import build_ext
# end stupid hackery
# https://github.com/cvondrick/pyvision/blob/07604f4445683365c5bee57a2276aebe05c244d4/setup.py

ext_modules=[Extension("timesegment._timesegment",
              [os.path.join("timesegment", "_timesegment", "_timesegment.pyx")],
              include_dirs=[numpy.get_include()])]

extensions = cythonize(ext_modules)

setup(
    # Packaging
    name='timesegment',
    version='0.1',
    description='Segment a timeseries using regression tree',
    long_description = 'TODO : write some long description',
    url='https://github.com/oboulant/timesegment',
    author='Olivier Boulant',
    author_email='olivier.boulant@telecom-paristech.org',
    license='BSD 3-clause "New" or "Revised" License',
    packages=['timesegment'],
    package_dir={'simplerandom' : 'timesegment'},
    install_requires = ['numpy'],
    classifiers = ['Programming Language :: Python :: 3',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Topic :: Scientific/Engineering'],
    keywords='time segment tree',
    # Cython
    cmdclass = {'build_ext': build_ext},
    ext_modules =  extensions
)
