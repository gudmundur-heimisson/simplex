from setuptools import setup

setup(
    name='linprog',
    author='Gudmundur Heimisson',
    version='0.1',
    package_dir={'': 'src'},
    packages=['linprog'],
    install_requires=['numpy'],
    setup_requires=['nose',
                    'wheel']
)
