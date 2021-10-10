from setuptools import find_packages, setup

with open("README.rst", "r") as fh:
    long_description = fh.read()

setup(
    name='pyyellin',
    version='0.1.0dev',
    author='Leonie Einfalt, '
           'Yasar Ilhan, '
           'Fatih Okçu, '
           'Felix Wagner',
    author_email="felix.wagner@oeaw.ac.at",
    description='A Python implementation of Yellins optimum interval method.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fewagner/pyyellin",
    license='GPLv3',
    packages=find_packages(include=['pyyellin', 'pyyellin.*']),
    install_requires=['numpy',
                      'matplotlib',
                      'scipy',
                      'tqdm',
                      # TODO add your required libraries here
                      ],
    # setup_requires=['pytest-runner'],
    # tests_require=['pytest==4.4.1'],
    test_suite='tests',
    classifiers=[
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
