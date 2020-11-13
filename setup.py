# -*- coding: utf-8 -*-
from setuptools import setup
from setuptools import find_packages

long_description = '''
This package is a snipped of the internal anysma project to train LVQ-based 
networks on GPU with Tensorflow. The package is sparsely documented and I 
highly recommend to study the examples first. There is no guarantee that the 
package works with other package version. Thus, I made the package versions 
strict and recommend the installation inside a docker or a pipenv. The 
required Tensorflow version requires Python 3.6.
'''

setup(name='anysma',
      version='0.0.0',
      description='Experiments of my PhD thesis.',
      long_description=long_description,
      author='Sascha Saralajew',
      author_email='sascha.saralajew@gmail.com',
      url='https://github.com/saralajew/thesis_GTLVQ_experiments',
      download_url='https://github.com/saralajew/thesis_GTLVQ_experiments.git',
      license='BSD 3-Clause License',
      install_requires=['tensorflow-gpu==2.3.1',
                        'keras==2.2.4',
                        'numpy==1.16.4',
                        'six==1.12.0',
                        'scikit-image==0.15.0',
                        'scikit-learn==0.21.2',
                        'matplotlib==3.1.0',
                        'scipy==1.3.0',
                        'sklearn==0.0',
                        'spectral==0.19',
                        'opencv-python==4.1.0.25'
                        ],
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())
