import sys
import os
from setuptools import setup, find_packages
from pip.req import parse_requirements

if sys.version_info.major < 3:
    print("I'm only for python 3, please upgrade")
    sys.exit(1)

install_reqs = parse_requirements(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'requirements.txt'),
                                  session=False)
reqs = [str(ir.req) for ir in install_reqs]

setup(
    name='jobtitle_classifier',
    version='0.1.dev1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=reqs,
    description="classify job titles into categories using machine learning",
    long_description="""\
    This package trains one of several machine learning models and can use them to translate job titles into classifications
    """,
    author="Michael Gooch",
    author_email="goochmi@gmail.com",
    url="https://bitbucket.org/kotoroshinoto/scribebinclustering",
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
    ],
    entry_points={
        'console_scripts': [
            'classify-jobtitles = scribe_classifier.cli.main:canada_model_cli'
        ]
    }
)
