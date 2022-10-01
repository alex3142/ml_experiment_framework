import setuptools

with open('requirements.txt') as f:
    req = [row.replace('\n', '') for row in f.readlines()]

setuptools.setup(
    name='cwt',
    version='0.1',
    packages=setuptools.find_packages(),
    install_requires=req
)