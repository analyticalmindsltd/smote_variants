import os
import codecs

from setuptools import setup

def readme():
    with codecs.open('README.rst', encoding='utf-8-sig') as f:
        return f.read()

version_file= os.path.join('smote_variants', '_version.py')
with open(version_file) as f:
    exec(f.read())

DISTNAME= 'smote_variants'
DESCRIPTION= 'Variants of the synthetic minority oversampling technique (SMOTE) for imbalanced learning'
LONG_DESCRIPTION= readme()
MAINTAINER= 'Gyorgy Kovacs'
MAINTAINER_EMAIL= 'gyuriofkovacs@gmail.com'
URL= 'https://github.com/gykovacs/smote-variants'
LICENSE= 'MIT'
DOWNLOAD_URL= 'https://github.com/gykovacs/smote-variants'
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
INSTALL_REQUIRES= ['numpy>=1.8.2', 'scipy>=0.13.3', 'scikit-learn>=0.20', 'joblib', 'minisom', 'statistics', 'tensorflow', 'keras']
EXTRAS_REQUIRE= {'tests': ['nose'],
                 'docs': ['sphinx', 'sphinx-gallery', 'sphinx_rtd_theme', 'matplotlib', 'pandas']}
PYTHON_REQUIRES= '>=3.5'
TEST_SUITE='nose.collector'
PACKAGE_DIR= {'smote_variants': 'smote_variants'}

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      zip_safe=False,
      classifiers=CLASSIFIERS,
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE,
      python_requires=PYTHON_REQUIRES,
      test_suite=TEST_SUITE,
      package_dir=PACKAGE_DIR)

#setup(name='smote_variants',
#      version=getversion(),
#      description='smote_variants',
#      long_description=readme(),
#      classifiers=[
#              'Development Status :: 3 - Alpha',
#              'License :: OSI Approved :: MIT License',
#              'Programming Language :: Python',
#              'Topic :: Scientific/Engineering :: Artificial Intelligence'],
#      url='http://github.com/gykovacs/smote_variants',
#      author='Gyorgy Kovacs',
#      author_email='gyuriofkovacs@gmail.com',
#      license='MIT',
#      packages=['smote_variants'],
#      install_requires=[
#              'joblib',
#              'numpy',
#              'pandas',
#              'scipy',
#              'sklearn',
#              'minisom',
#              'statistics',
#              ],
#      py_modules=['smote_variants'],
#      python_requires='>=3.5',
#      zip_safe=False,
#      package_dir= {'smote_variants': 'smote_variants'},
#      package_data= {},
#      tests_require= ['nose'],
#      test_suite= 'nose.collector'
#      )
