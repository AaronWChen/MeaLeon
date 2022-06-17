""" This script is to be loaded by notebooks to incorporate custom scripts and classes from the [base]/src/ folder. It is intended to be included within the notebooks/exploratory/ and notebooks/final/ folders. 

Import this like a normal script BUT not in lexicographical order. It must be imported first in order to make sure that custom scripts can be imported.
"""

import os
import sys

module_path = os.path.abspath(os.path.join(os.pardir, os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)