# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "TinyStan"
year = datetime.date.today().year
copyright = f"{year}, TinyStan Developers"
author = "TinyStan Developers"

import os

import tinystan

most_recent_release = "v" + tinystan.__version__
version = os.getenv("TS_DOCS_VERSION", most_recent_release)
if version == "latest":
    # don't display a version number for "latest" docs
    switcher_version = "latest"
    release = ""
else:
    release = version.strip("v")
    switcher_version = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx_copybutton",
    "nbsphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
]

myst_enable_extensions = ["substitution"]
myst_substitutions = {"most_recent_release": most_recent_release}

suppress_warnings = ["myst.xref_missing"]  # Julia doc generates raw html links

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md", "languages/_*"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = [
    "css/Documenter.css",
    "css/custom.css",
]
# html_favicon = "_static/image/favicon.ico"

html_show_sphinx = False

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/WardBrian/tinystan",
            "icon": "fab fa-github",
        },
        {
            "name": "Forums",
            "url": "https://discourse.mc-stan.org/",
            "icon": "fas fa-users",
        },
    ],
    "use_edit_page_button": True,
    "switcher": {
        "json_url": "https://raw.githubusercontent.com/WardBrian/tinystan/main/docs/_static/switcher.json",
        "version_match": switcher_version,
    },
    "logo": {
        "text": "TinyStan " + release if release else "TinyStan",
        # "image_light": "_static/image/icon.png",
        # "image_dark": "_static/image/icon_w.png",
        "alt_text": "TinyStan - Home",
    },
    "navbar_end": ["theme-switcher", "navbar-icon-links", "version-switcher"],
}

html_context = {
    "github_user": "WardBrian",
    "github_repo": "tinystan",
    "github_version": "main",
    "doc_path": "docs",
}

latex_logo = "_static/image/logo.pdf"

intersphinx_mapping = {
    "python": (
        "https://docs.python.org/3/",
        None,
    ),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "cmdstanpy": ("https://mc-stan.org/cmdstanpy/", None),
    "bridgestan": ("https://roualdes.github.io/bridgestan/latest/", None),
}


breathe_projects = {"tinystan": "./_build/cppxml/"}
breathe_projects_source = {"tinystan": ("../src/", ["tinystan.h"])}
breathe_default_project = "tinystan"
# doxygen doesn't like  __attribute and __declspec
# https://www.doxygen.nl/manual/preprocessing.html
breathe_doxygen_config_options = {
    "ENABLE_PREPROCESSING": "YES",
    "MACRO_EXPANSION": "YES",
    "EXPAND_ONLY_PREDEF": "YES",
    "PREDEFINED": "TINYSTAN_PUBLIC=",
}

autoclass_content = "both"

import os
import subprocess
import pathlib

RUNNING_IN_CI = os.environ.get("CI") or os.environ.get("READTHEDOCS")
BASE_DIR = pathlib.Path(__file__).parent.parent

try:
    print("Building Julia doc")
    subprocess.run(
        ["julia", "--project=.", "./make.jl"],
        cwd=BASE_DIR / "clients" / "julia" / "docs",
        check=True,
    )
except Exception as e:
    # fail loudly in Github Actions
    if RUNNING_IN_CI:
        raise e
    else:
        print("Failed to build julia docs!\n", e)

try:
    print("Building R doc")
    subprocess.run(
        ["Rscript", "convert_docs.R"],
        cwd=BASE_DIR / "clients" / "R",
        check=True,
    )

except Exception as e:
    # fail loudly in Github Actions
    if RUNNING_IN_CI:
        raise e
    else:
        print("Failed to build R docs!\n", e)


try:
    print("Checking C++ doc availability")
    import breathe

    subprocess.run(["doxygen", "-v"], check=True, capture_output=True)
except Exception as e:
    if RUNNING_IN_CI:
        raise e
    else:
        print("Breathe/doxygen not installed, skipping C++ Doc")
        exclude_patterns += ["languages/c-api.rst"]
else:
    extensions.append("breathe")

try:
    print("Building JS doc")
    yarn = os.getenv("YARN", "yarn").split()
    ret = subprocess.run(
        yarn + ["--silent", "doc"],
        cwd=BASE_DIR / "clients" / "typescript",
        check=True,
        capture_output=True,
        text=True,
    )
    with open("./languages/js.md", "w") as f:
        f.write(ret.stdout)

except Exception as e:
    # fail loudly in Github Actions
    if RUNNING_IN_CI:
        raise e
    else:
        print("Failed to build JS docs!\n", e)
