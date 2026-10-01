"""Shared code for the notebooks.

On Windows, TensorFlow's native runtime fails to load if pandas, pyarrow or
shap were imported first, so it is imported here, before anything else. Import
`src` before other data libraries in every notebook.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")   # deterministic CPU kernels

import tensorflow  # noqa: E402,F401  (must be loaded first; see docstring)
