from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='smote_variants',
      version='0.1.2',
      description='smote_variants',
      long_description=readme(),
      classifiers=[
              'Development Status :: 3 - Alpha',
              'License :: OSI Approved :: MIT License',
              'Programming Language :: Python',
              'Topic :: Scientific/Engineering :: Artificial Intelligence'],
      url='http://github.com/gykovacs/smote_variants',
      author='Gyorgy Kovacs',
      author_email='gyuriofkovacs@gmail.com',
      license='MIT',
      packages=['smote_variants'],
      install_requires=[
              'joblib',
              'numpy',
              'pandas',
              'scipy',
              'sklearn',
              'minisom',
              'statistics',
              ],
      py_modules=['smote_variants'],
      python_requires='>=3.5',
      zip_safe=False,
      package_dir= {'smote_variants': 'smote_variants'},
      package_data= {},
      tests_require= ['nose'],
      test_suite= 'nose.collector'
      )
