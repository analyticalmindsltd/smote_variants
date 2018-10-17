from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='smote_variants',
      version='0.1',
      description='smote_variants',
      long_description=readme(),
      classifiers=[
              'Development Status :: 3 - Alpha',
              'License :: MIT License',
              'Programming Language :: Python',
              'Topic :: Machine Learning'],
      url='http://github.com/gykovacs/smote_variants',
      author='Gyorgy Kovacs',
      author_email='gyuriofkovacs@gmail.com',
      license='MIT',
      packages=['smote_variants'],
      install_requires=[
              'numpy',
              'pandas',
              'scipy',
              'sklearn',
              'minisom',
              'statistics',
              ],
      py_modules=['smote_variants'],
      zip_safe=False,
      package_dir= {'smote_variants': 'smote_variants'},
      package_data= {},
      tests_require= ['nose'],
      test_suite= 'nose.collector'
      )
