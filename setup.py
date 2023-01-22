from pkg_resources import parse_version
from configparser import ConfigParser
import setuptools
assert parse_version(setuptools.__version__)>=parse_version('36.2')

# note: all settings are in settings.ini; edit there, not here
config = ConfigParser(delimiters=['='])
config.read('settings.ini')
cfg = config['DEFAULT']

cfg_keys = 'version description keywords author author_email'.split()
expected = cfg_keys + "lib_name user branch license status min_python audience language".split()
for o in expected: assert o in cfg, "missing expected setting: {}".format(o)
setup_cfg = {o:cfg[o] for o in cfg_keys}

licenses = {
    'apache2': ('Apache Software License 2.0','OSI Approved :: Apache Software License'),
    'mit': ('MIT License', 'OSI Approved :: MIT License'),
    'gpl2': ('GNU General Public License v2', 'OSI Approved :: GNU General Public License v2 (GPLv2)'),
    'gpl3': ('GNU General Public License v3', 'OSI Approved :: GNU General Public License v3 (GPLv3)'),
    'bsd3': ('BSD License', 'OSI Approved :: BSD License'),
}
statuses = [ '1 - Planning', '2 - Pre-Alpha', '3 - Alpha',
    '4 - Beta', '5 - Production/Stable', '6 - Mature', '7 - Inactive' ]
py_versions = '3.6 3.7 3.8 3.9 3.10'.split()

# requirements = cfg.get('requirements','').split()
install_requires = \
['Flask>=2.0.2,<3.0.0',
 'adjustText>=0.7.3,<0.8.0',
 'bokeh>=2.4.3,<3.0.0',
 'gensim>=4.1.2,<5.0.0',
 'graphviz>=0.20.1,<0.21.0',
 'gunicorn>=20.1.0,<21.0.0',
 'hdbscan>=0.8.29,<0.9.0',
 'importlib-metadata>=4.12.0,<5.0.0',
 'importlib-resources>=5.8.0,<6.0.0',
 'jupyter>=1.0.0,<2.0.0',
 'jupyterlab>=3.2.5,<4.0.0',
 'matplotlib>=3.5.2,<4.0.0',
 'nltk>=3.7,<4.0',
 'numpy>=1.21.5,<2.0.0',
 'openTSNE>=0.6.2,<0.7.0',
 'optuna>=3.0.3,<4.0.0',
 'pandas>=1.3.5,<2.0.0',
 'python-utils>=3.3.3,<4.0.0',
 'requests>=2.26.0,<3.0.0',
 'scikit-learn>=1.0.1,<2.0.0',
 'scipy>=1.7.3,<2.0.0',
 'seaborn>=0.11.2,<0.12.0',
 'spacy>=3.2.1,<4.0.0',
 'stanza>=1.3.0,<2.0.0',
 'tqdm>=4.62.3,<5.0.0',
 'umap-learn>=0.5.3,<0.6.0',
 'waitress>=2.1.2,<3.0.0',
 'xgboost>=1.6.1,<2.0.0',
 'zipp>=3.8.0,<4.0.0']
# if cfg.get('pip_requirements'): requirements += cfg.get('pip_requirements','').split()
min_python = cfg['min_python']
lic = licenses.get(cfg['license'].lower(), (cfg['license'], None))
dev_requirements = (cfg.get('dev_requirements') or '').split()

setuptools.setup(
    name = cfg['lib_name'],
    license = lic[0],
    classifiers = [
        'Development Status :: ' + statuses[int(cfg['status'])],
        'Intended Audience :: ' + cfg['audience'].title(),
        'Natural Language :: ' + cfg['language'].title(),
    ] + ['Programming Language :: Python :: '+o for o in py_versions[py_versions.index(min_python):]] + (['License :: ' + lic[1] ] if lic[1] else []),
    url = cfg['git_url'],
    packages = setuptools.find_packages(),
    include_package_data = True,
    install_requires = requirements,
    extras_require={ 'dev': dev_requirements },
    dependency_links = cfg.get('dep_links','').split(),
    python_requires  = '>=' + cfg['min_python'],
    long_description = open('README.md').read(),
    long_description_content_type = 'text/markdown',
    zip_safe = False,
    entry_points = {
        'console_scripts': cfg.get('console_scripts','').split(),
        'nbdev': [f'{cfg.get("lib_path")}={cfg.get("lib_path")}._modidx:d']
    },
    **setup_cfg)


