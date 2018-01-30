import pyhivemall
VERSION = pyhivemall.__version__

DISTNAME = 'pyhivemall'
DESCRIPTION = 'Using machine learning model from Apache Hivemall in Python'
LONG_DESCRIPTION = __doc__ or ''
AUTHOR = 'Takuya Kitazawa'
AUTHOR_EMAIL = 'k.takuti@gmail.com'
MAINTAINER = AUTHOR
MAINTAINER_EMAIL = AUTHOR_EMAIL
LICENSE = 'Apache License 2.0'
URL = 'https://github.com/takuti/pyhivemall'
DOWNLOAD_URL = ''


def setup_package():
    from setuptools import setup, find_packages

    metadata = dict(
        name=DISTNAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        license=LICENSE,
        url=URL,
        download_url=DOWNLOAD_URL,
        classifiers=['License :: OSI Approved :: Apache Software License',
                     'Programming Language :: Python :: 2',
                     'Programming Language :: Python :: 2.7',
                     'Programming Language :: Python :: 3',
                     'Programming Language :: Python :: 3.4',
                     'Programming Language :: Python :: 3.5',
                     'Programming Language :: Python :: 3.6'],
        packages=find_packages(exclude=['*tests*']),
        install_requires=[
            'numpy',
            'scikit_learn',
            'pandas-td',
            'pyhive',
            'thrift',
            'thrift-sasl'])

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
