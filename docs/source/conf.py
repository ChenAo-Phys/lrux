# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import pathlib
import sys

sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

project = "lrux"
copyright = "2025, Ao Chen, Christopher Roth"
author = "Ao Chen, Christopher Roth"
release = "0.1.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.autosummary"]

templates_path = ["_templates"]
exclude_patterns = []

autodoc_member_order = "bysource"
autosummary_generate = True
autosummary_generate_overwrite = False
default_role = "py:obj"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
# html_static_path = ["_static"]
html_use_relative_urls = True
html_baseurl = "https://chenao-phys.github.io/lrux/"

html_theme_options = {
    "external_links": [],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/ChenAo-Phys/lrux/tree/main",
            "icon": "fab fa-github",  # FontAwesome icon
        },
    ],
}
