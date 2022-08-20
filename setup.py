import os
import codecs

from setuptools import setup, find_packages

def readme():
    with codecs.open('README.rst', encoding='utf-8-sig') as f:
        return f.read()

version_file= os.path.join('smote_variants', '_version.py')
__version__= "0.0.0"
with open(version_file) as f:
    exec(f.read())

from setuptools.command.test import test as TestCommand
import sys

class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        #import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)

DISTNAME= 'smote_variants'
DESCRIPTION= 'Variants of the synthetic minority oversampling technique (SMOTE) for imbalanced learning'
LONG_DESCRIPTION= readme()
LONG_DESCRIPTION_CONTENT_TYPE='text/x-rst'
MAINTAINER= 'Gyorgy Kovacs'
MAINTAINER_EMAIL= 'gyuriofkovacs@gmail.com'
URL= 'https://github.com/analyticalmindsltd/smote_variants'
LICENSE= 'MIT'
DOWNLOAD_URL= 'https://github.com/analyticalmindsltd/smote_variants'
VERSION= __version__
CLASSIFIERS= [  'Intended Audience :: Science/Research',
                'Intended Audience :: Developers',
                'Development Status :: 3 - Alpha',
                'License :: OSI Approved :: MIT License',
                'Programming Language :: Python',
                'Topic :: Scientific/Engineering :: Artificial Intelligence',
                'Topic :: Software Development',
                'Operating System :: Microsoft :: Windows',
                'Operating System :: POSIX',
                'Operating System :: Unix',
                'Operating System :: MacOS']
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    INSTALL_REQUIRES = []
else:
    INSTALL_REQUIRES= ['numpy>=1.13.0', 'scipy', 'scikit-learn', 'joblib', 'minisom', 'statistics', 'tensorflow', 'keras', 'pandas', 'mkl', 'metric_learn', 'seaborn']
EXTRAS_REQUIRE= {'tests': ['pytest'],
                 'docs': ['sphinx', 'sphinx-gallery', 'sphinx_rtd_theme', 'matplotlib', 'pandas']}
PYTHON_REQUIRES= '>=3.5'
CMDCLASS = {'test': PyTest}
PACKAGE_DIR= {'smote_variants': 'smote_variants'}
SETUP_REQUIRES=['setuptools>=41.0.1', 'wheel>=0.33.4', 'pytest-runner']
TESTS_REQUIRE=['pytest']

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      author=MAINTAINER,
      author_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
      zip_safe=False,
      classifiers=CLASSIFIERS,
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE,
      python_requires=PYTHON_REQUIRES,
      setup_requires=SETUP_REQUIRES,
      tests_require=TESTS_REQUIRE,
      cmdclass=CMDCLASS,
      package_dir=PACKAGE_DIR,
      packages=find_packages(exclude=[]))
