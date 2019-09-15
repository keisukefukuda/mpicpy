import os.path
from setuptools import setup
import importlib.machinery as imm
import configparser

project_dir = os.path.dirname(__file__)

# Version
version_py = os.path.join(project_dir, 'mpicpy', 'version.py')
version = imm.SourceFileLoader('version', version_py).load_module()

# Dependencies
config = configparser.ConfigParser()
config.read(os.path.join(project_dir, 'Pipfile'))


setup(
    name='mpicpy',
    version='{}.{}'.format(version.VERSION_MAJOR, version.VERSION_MINOR),
    install_requires=list(config['packages'].keys()),
    extras_require = {
        'develop': list(config['dev-packages'].keys())
    },
    entry_points = {
        "console_scripts": [
            "mpicpy = mpicpy:main"
        ]
    }
)