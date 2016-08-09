import glob
import os
import re

file_names = glob.glob("*.tif")
for file_name in file_names:
	groups = re.split('(\d+)', file_name)
	new_file_name = groups[0] + " z_" + groups[1] + ".tif"
	print file_name, new_file_name
	os.rename(file_name, new_file_name)
