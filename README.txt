Put the folder amrvac_tools in the amrvac source directory, under
/amrvac/tools/python

Mac/Linux: add this to PYTHONPATH in ~/.bashrc

export PYTHONPATH="${PYTHONPATH}:../../amrvac_tools" (replace ../.. by the path to the amrvac_tools folder)

OR

add this at the beginning of any python script:

import sys
sys.path.append('path to amrvac_tools')