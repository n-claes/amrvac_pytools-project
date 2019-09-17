Adding the folder 'amrvac_tools' to the Python path:

Windows users: Start -> control panel -> 'edit the system environment variables' -> tab 'hardware' -> 'environment variables'
	       search for PYTHONPATH (or create one if it is not already there) and add the path to the 'amrvac_tools' folder

Mac & Linux users can add the following line to their PYTHONPATH in ~/.bashprofile (Mac) or ~/.bashrc (Linux)

                   export PYTHONPATH="${PYTHONPATH}:../../amrvac_tools"

where you replace '../..' with the actual path containing the 'amrvac_tools' folder.

--------------------------------------------------------------

Users that not want to modify their PYTHONPATH, can add these two lines at the beginning of every Python script where the tools are to be used:

                   import sys
                   sys.path.append('path to amrvac_tools')

where you replace 'path to amrvac_tools' with the actual path containing the 'amrvac_tools' folder.