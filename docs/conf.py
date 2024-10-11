# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Gammapy documentation build configuration file.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# Note that not all possible configuration values are present in this file.
#
# All configuration values have a default. Some values are defined in
# the global Astropy configuration which is loaded here before anything else.
# See astropy.sphinx.conf for which values are set there.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath('..'))
# IMPORTANT: the above commented section was generated by sphinx-quickstart, but
# is *NOT* appropriate for astropy or Astropy affiliated packages. It is left
# commented out with this explanation to make it clear why this should not be
# done. If the sys.path entry above is added, when the astropy.sphinx.conf
# import occurs, it will import the *source* version of astropy instead of the
# version installed (if invoked as "make html" or directly with sphinx), or the
# version in the build directory (if "python setup.py build_sphinx" is used).
# Thus, any C-extensions that are needed to build the documentation will *not*
# be accessible, and the documentation will not build correctly.

import datetime
import sys
import os

# Get configuration information from setup.cfg
from configparser import ConfigParser
from pkg_resources import get_distribution

# Load all the global Astropy configuration
from sphinx_astropy.conf import *

# Sphinx-gallery config
from sphinx_gallery.sorting import ExplicitOrder

# Load utils docs functions
from gammapy.utils.docs import SubstitutionCodeBlock, gammapy_sphinx_ext_activate

# flake8: noqa


# Add our custom directives to Sphinx
def setup(app):
    """
    Add the custom directives to Sphinx.
    """
    app.add_config_value("substitutions", [], "html")
    app.add_directive("substitution-code-block", SubstitutionCodeBlock)


conf = ConfigParser()
conf.read([os.path.join(os.path.dirname(__file__), "..", "setup.cfg")])
setup_cfg = dict(conf.items("metadata"))

sys.path.insert(0, os.path.dirname(__file__))

linkcheck_anchors_ignore = []
linkcheck_ignore = [
    "http://gamma-sky.net/#",
    "https://bitbucket.org/hess_software/hess-open-source-tools/src/master/",
    "https://forge.in2p3.fr/projects/data-challenge-1-dc-1/wiki",
    "https://indico.cta-observatory.org/event/2070/",
    "https://data.hawc-observatory.org/datasets/3hwc-survey/index.php",
    "https://github.com/gammapy/gammapy#status-shields",
    "https://groups.google.com/forum/#!forum/astropy-dev",
    "https://lists.nasa.gov/mailman/listinfo/open-gamma-ray-astro",
    "https://getbootstrap.com/css/#tables",
    "https://www.hawc-observatory.org/",  # invalid certificate
    "https://ipython.org",  # invalid certificate
    "https://jupyter.org",  # invalid certificate
    "https://hess-confluence.desy.de/confluence/display/HESS/HESS+FITS+data", # private page
    "https://hess-confluence.desy.de/"
]

# the buttons link to html pages which are auto-generated...
linkcheck_exclude_documents = [r"getting-started/.*"]

# -- General configuration ----------------------------------------------------

# By default, highlight as Python 3.
highlight_language = "python3"

# Matplotlib directive sets whether to show a link to the source in HTML
plot_html_show_source_link = False

# If true, figures, tables and code-blocks are automatically numbered if they have a caption
numfig = False

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = "1.1"

# We currently want to link to the latest development version of the astropy docs,
# so we override the `intersphinx_mapping` entry pointing to the stable docs version
# that is listed in `astropy/sphinx/conf.py`.
intersphinx_mapping.pop("h5py", None)
intersphinx_mapping["matplotlib"] = ("https://matplotlib.org/", None)
intersphinx_mapping["astropy"] = ("http://docs.astropy.org/en/latest/", None)
intersphinx_mapping["regions"] = (
    "https://astropy-regions.readthedocs.io/en/latest/",
    None,
)
intersphinx_mapping["reproject"] = ("https://reproject.readthedocs.io/en/latest/", None)
intersphinx_mapping["naima"] = ("https://naima.readthedocs.io/en/latest/", None)
intersphinx_mapping["gadf"] = (
    "https://gamma-astro-data-formats.readthedocs.io/en/latest/",
    None,
)
intersphinx_mapping["iminuit"] = ("https://iminuit.readthedocs.io/en/latest/", None)
intersphinx_mapping["pandas"] = ("https://pandas.pydata.org/pandas-docs/stable/", None)

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns.append("_templates")
exclude_patterns.append("**.ipynb_checkpoints")
exclude_patterns.append("user-guide/model-gallery/*/*.ipynb")
exclude_patterns.append("user-guide/model-gallery/*/*.md5")
exclude_patterns.append("user-guide/model-gallery/*/*.py")

extensions.extend(
    [
        "sphinx_click.ext",
        "sphinx.ext.mathjax",
        "sphinx_gallery.gen_gallery",
        "sphinx.ext.doctest",
        "sphinx_design",
        "sphinx_copybutton",
        "sphinx_automodapi.smart_resolver",
    ]
)

nbsphinx_execute = "never"
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
# --

# This is added to the end of RST files - a good place to put substitutions to
# be used globally.
rst_epilog += """
.. |Table| replace:: :class:`~astropy.table.Table`
"""

# This is added to keep the links to PRs in release notes
changelog_links_docpattern = [".*changelog.*", "whatsnew/.*", "release-notes/.*"]

# -- Project information ------------------------------------------------------

# This does not *have* to match the package name, but typically does
project = setup_cfg["name"]
author = setup_cfg["author"]
copyright = "{}, {}".format(datetime.datetime.now().year, setup_cfg["author"])

version = get_distribution(project).version
release = "X.Y.Z"
switch_version = version
if "dev" in version:
    switch_version = "dev"
else:
    release = version

substitutions = [
    ("|release|", release),
]
# -- Options for HTML output ---------------------------------------------------

# A NOTE ON HTML THEMES
# The global astropy configuration uses a custom theme, "bootstrap-astropy",
# which is installed along with astropy. A different theme can be used or
# the options for this theme can be modified by overriding some
# variables set in the global configuration. The variables set in the
# global configuration are listed below, commented out.

# Add any paths that contain custom themes here, relative to this directory.
# To use a different custom theme, add the directory containing the theme.
# html_theme_path = []

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes. To override the custom theme, set this to the
# name of a builtin theme or the name of a custom theme in html_theme_path.
html_theme = "pydata_sphinx_theme"

# Static files to copy after template files
html_static_path = ["_static"]

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_logo = os.path.join(html_static_path[0], "gammapy_logo_nav.png")
html_favicon = os.path.join(html_static_path[0], "gammapy_logo.ico")

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    "search": ["search-field.html"],
    "navigation": ["sidebar-nav-bs.html"],
}

# If not "", a "Last updated on:" timestamp is inserted at every page bottom,
# using the given strftime format.
# html_last_updated_fmt = ""

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "{} v{}".format(project, release)

# Output file base name for HTML help builder.
htmlhelp_basename = f"{project}doc"

html_theme_options = {
    # toc options
    "collapse_navigation": False,
    "navigation_depth": 2,
    "show_prev_next": False,
    # links in menu
    "icon_links": [
        {
            "name": "Github",
            "url": "https://github.com/gammapy/gammapy",
            "icon": "fab fa-github-square",
        },
        {
            "name": "Twitter",
            "url": "https://twitter.com/gammapyST",
            "icon": "fab fa-twitter-square",
        },
        {
            "name": "Slack",
            "url": "https://gammapy.slack.com/",
            "icon": "fab fa-slack",
        },
    ],
    "switcher": {
        "json_url": "https://docs.gammapy.org/stable/switcher.json",
        "version_match": switch_version,
    },
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    "navigation_with_keys": True,
    # footers
    "footer_start": ["copyright"],
    "footer_center": ["last-updated"],
    "footer_end": ["sphinx-version", "theme-version"]
}


gammapy_sphinx_ext_activate()

# -- Options for LaTeX output --------------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
    ("index", f"{project}.tex", f"{project} Documentation", author, "manual")
]

# -- Options for manual page output --------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [("index", project.lower(), f"{project} Documentation", [author], 1)]


# -- Other options --

github_issues_url = "https://github.com/gammapy/gammapy/issues/"

# http://sphinx-automodapi.readthedocs.io/en/latest/automodapi.html
# show inherited members for classes
automodsumm_inherited_members = True

# In `about.rst` and `references.rst` we are giving lists of citations
# (e.g. papers using Gammapy) that partly aren"t referenced from anywhere
# in the Gammapy docs. This is normal, but Sphinx emits a warning.
# The following config option suppresses the warning.
# http://www.sphinx-doc.org/en/stable/rest.html#citations
# http://www.sphinx-doc.org/en/stable/config.html#confval-suppress_warnings
suppress_warnings = ["ref.citation"]

branch = "main" if switch_version == "dev" else f"v{switch_version}"

binder_config = {
    # Required keys
    "org": "gammapy",
    "repo": "gammapy-webpage",
    "branch": branch,  # Can be any branch, tag, or commit hash. Use a branch that hosts your docs.
    "binderhub_url": "https://mybinder.org",  # Any URL of a binderhub deployment. Must be full URL (e.g. https://mybinder.org).
    "dependencies": "./binder/requirements.txt",
    "notebooks_dir": f"notebooks/{switch_version}",
    "use_jupyter_lab": True,
}

# nitpicky = True
sphinx_gallery_conf = {
    "examples_dirs": [
        "../examples/models",
        "../examples/tutorials",
    ],  # path to your example scripts
    "gallery_dirs": [
        "user-guide/model-gallery",
        "tutorials",
    ],  # path to where to save gallery generated output
    "subsection_order": ExplicitOrder(
        [
            "../examples/models/spatial",
            "../examples/models/spectral",
            "../examples/models/temporal",
            "../examples/tutorials/starting",
            "../examples/tutorials/data",
            "../examples/tutorials/analysis-1d",
            "../examples/tutorials/analysis-2d",
            "../examples/tutorials/analysis-3d",
            "../examples/tutorials/analysis-time",
            "../examples/tutorials/api",
            "../examples/tutorials/scripts",
        ]
    ),
    "binder": binder_config,
    "backreferences_dir": "gen_modules/backreferences",
    "doc_module": ("gammapy",),
    "exclude_implicit_doc": {},
    "filename_pattern": r"\.py",
    "reset_modules": ("matplotlib",),
    "within_subsection_order": "sphinxext.TutorialExplicitOrder",
    "download_all_examples": True,
    "capture_repr": ("_repr_html_", "__repr__"),
    "nested_sections": False,
    "min_reported_time": 10,
    "show_memory": False,
    "line_numbers": False,
    "reference_url": {
        # The module you locally document uses None
        "gammapy": None,
    },
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["matomo.js"]

html_context = {
    "default_mode": "light",
}

# Add-on to insert the Matomo tracker
templates_path = ['_templates']
