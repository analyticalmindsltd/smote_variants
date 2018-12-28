#!/bin/bash

python ./setup.py register sdist bdist_wheel
twine upload --verbose --repository-url https://upload.pypi.org/legacy/ dist/*

