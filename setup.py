#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import glob
import os
import sys

import ah_bootstrap
from setuptools import setup

# A dirty hack to get around some early import/configurations ambiguities
if sys.version_info[0] >= 3:
    import builtins
else:
    import __builtin__ as builtins
builtins._ASTROPY_SETUP_ = True

from astropy_helpers.setup_helpers import (
    register_commands, adjust_compiler, get_debug_option, get_package_info)
from astropy_helpers.git_helpers import get_git_devstr
from astropy_helpers.version_helpers import generate_version_py

# Get some values from the setup.cfg
from distutils import config
conf = config.ConfigParser()
conf.read(['setup.cfg'])
metadata = dict(conf.items('metadata'))

PACKAGENAME = metadata.get('package_name', 'packagename')
DESCRIPTION = metadata.get('description', 'Astropy affiliated package')
AUTHOR = metadata.get('author', '')
AUTHOR_EMAIL = metadata.get('author_email', '')
LICENSE = metadata.get('license', 'unknown')
URL = metadata.get('url', 'http://astropy.org')

# Get the long description from the package's docstring
#__import__(PACKAGENAME)
#package = sys.modules[PACKAGENAME]
#LONG_DESCRIPTION = package.__doc__
LONG_DESCRIPTION = open('LONG_DESCRIPTION.rst').read()


# Store the package name in a built-in variable so it's easy
# to get from other parts of the setup infrastructure
builtins._ASTROPY_PACKAGE_NAME_ = PACKAGENAME

# VERSION should be PEP386 compatible (http://www.python.org/dev/peps/pep-0386)
# We use the format is `x.y` or `x.y.z` or `x.y.dev`
VERSION = '0.4.dev'

# Indicates if this version is a release version
RELEASE = 'dev' not in VERSION

if not RELEASE:
    VERSION += get_git_devstr(False)

# Populate the dict of setup command overrides; this should be done before
# invoking any other functionality from distutils since it can potentially
# modify distutils' behavior.
cmdclassd = register_commands(PACKAGENAME, VERSION, RELEASE)

# Adjust the compiler in case the default on this platform is to use a
# broken one.
adjust_compiler(PACKAGENAME)

# Freeze build information in version.py
generate_version_py(PACKAGENAME, VERSION, RELEASE,
                    get_debug_option(PACKAGENAME))

# Get configuration information from all of the various subpackages.
# See the docstring for setup_helpers.update_package_files for more
# details.
package_info = get_package_info()

# Add the project-global data
package_info['package_data'].setdefault(PACKAGENAME, [])

# Command-line scripts
# Please keep the list in alphabetical order
entry_points = {}
entry_points['console_scripts'] = [
    'gammapy-bin-cube = gammapy.scripts.bin_cube:main',
    'gammapy-bin-image = gammapy.scripts.bin_image:main',
    'gammapy-catalog-browse = gammapy.scripts.catalog_browser:main',
    'gammapy-catalog-query = gammapy.scripts.catalog_query:main',
    'gammapy-coordinate-images = gammapy.scripts.coordinate_images:main',
    'gammapy-cwt = gammapy.scripts.cwt:main',
    'gammapy-data-manage = gammapy.obs.data_manager:main',
    'gammapy-data-browse = gammapy.scripts.data_browser:main',
    'gammapy-data-show = gammapy.scripts.data_show:main',
    'gammapy-derived-images = gammapy.scripts.derived_images:main',
    'gammapy-detect = gammapy.scripts.detect:main',
    'gammapy-image-decompose-a-trous = gammapy.scripts.image_decompose_a_trous:main',
    'gammapy-image-pipe = gammapy.scripts.image_pipe:main',
    'gammapy-info = gammapy.scripts.info:main',
    'gammapy-iterative-source-detect = gammapy.scripts.iterative_source_detect:main',
    'gammapy-look-up-image = gammapy.scripts.look_up_image:main',
    'gammapy-model-image = gammapy.scripts.model_image:main',
    'gammapy-make-bg-cube-models = gammapy.scripts.make_bg_cube_models:main',
    'gammapy-obs-select = gammapy.scripts.obs_select:main',
    'gammapy-pfmap = gammapy.scripts.pfmap:main',
    'gammapy-pfsim = gammapy.scripts.pfsim:main',
    'gammapy-pfspec = gammapy.scripts.pfspec:main',
    'gammapy-reflected-regions = gammapy.scripts.reflected_regions:main',
    'gammapy-residual-images = gammapy.scripts.residual_images:main',
    'gammapy-sherpa-like = gammapy.scripts.sherpa_like:main',
    'gammapy-sherpa-hspec = gammapy.hspec.run_fit:main',
    'gammapy-sherpa-model-image = gammapy.scripts.sherpa_model_image:main',
    'gammapy-significance-image = gammapy.scripts.significance_image:main',
    'gammapy-simulate-source-catalog = gammapy.scripts.simulate_source_catalog:main',
    'gammapy-test = gammapy.scripts.check:main',
    'gammapy-ts-image = gammapy.scripts.ts_image:main',
    'gammapy-spectrum = gammapy.spectrum.spectrum_analysis:main',
    'gammapy-spectrum-pipe = gammapy.scripts.spectrum_pipe:main',
]

# Note: usually the `affiliated_package/data` folder is used for data files.
# In Gammapy we use `gammapy/data` as a sub-package.
# Uncommenting the following line was needed to avoid an error during
# the `python setup.py build` phase
# package_info['package_data'][PACKAGENAME].append('data/*')

# Include all .c files, recursively, including those generated by
# Cython, since we can not do this in MANIFEST.in with a "dynamic"
# directory name.
c_files = []
for root, dirs, files in os.walk(PACKAGENAME):
    for filename in files:
        if filename.endswith('.c'):
            c_files.append(
                os.path.join(
                    os.path.relpath(root, PACKAGENAME), filename))
package_info['package_data'][PACKAGENAME].extend(c_files)

setup(name=PACKAGENAME,
      version=VERSION,
      description=DESCRIPTION,
      # Note: these are the versions we test.
      # Older versions could work, but are unsupported.
      # To find out if everything works run the Gammapy tests.
      install_requires=[
          'numpy>=1.6',
          'astropy>=1.0',
      ],
      extras_require=dict(
          analysis=[
              'click',
              'scipy>=0.14',
              'scikit-image>=0.10',
              'photutils>=0.1',
              'reproject',
              'gwcs',
              'astroplan',
              'uncertainties>=2.4',
              'naima',
              'iminuit',
              'sherpa',
          ],
          plotting=[
              'matplotlib>=1.4',
              'wcsaxes>=0.3',
              'aplpy>=0.9',
          ],
          gui=[
              'flask',
              'flask-bootstrap',
              'flask-wtf',
              'flask-nav',
          ],
      ),
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      url=URL,
      long_description=LONG_DESCRIPTION,
      classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Programming Language :: C',
          'Programming Language :: Cython',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: Implementation :: CPython',
          'Topic :: Scientific/Engineering :: Astronomy',
          'Development Status :: 3 - Beta',
      ],
      cmdclass=cmdclassd,
      zip_safe=False,
      use_2to3=False,
      entry_points=entry_points,
      **package_info
)
