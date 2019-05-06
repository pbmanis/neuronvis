from setuptools import setup, find_packages
import os

# Use Semantic Versioning, http://semver.org/
version_info = (0, 1, 0, '')
__version__ = '%d.%d.%d%s' % version_info


setup(name='neuronvis',
      version=__version__,
      description='Neuron Visualization module',
      url='http://github.com/pbmanis/neuronvis',
      author='Paul B. Manis and Luke Campagnola',
      author_email='pmanis@med.unc.edu',
      license='MIT',
      packages=find_packages(include=['neuronvis*']),
      install_requires=['matplotlib>=3.0', 'numpy>=1.1', 'mayavi'
          ],
      zip_safe=False,
      entry_points={
          'console_scripts': [
               'hocRender=neuronvis.hocRender:main',
               ],
          # 'gui_scripts': [
          #       'event_monger=src.event_monger:main',
          # ]
      },
      classifiers = [
             "Programming Language :: Python :: 3.6+",
             "Development Status ::  Beta",
             "Environment :: Console",
             "Intended Audience :: Manis Lab",
             "License :: MIT",
             "Operating System :: OS Independent",
             "Topic :: Software Development :: Tools :: Python Modules",
             "Topic :: Computational Modeling :: Neuroscience",
             ],
    )
      