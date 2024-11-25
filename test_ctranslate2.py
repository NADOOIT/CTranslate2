import os
os.environ["DYLD_LIBRARY_PATH"] = "/usr/local/lib"

import ctranslate2
print("Available device types:", ctranslate2.get_supported_compute_types())
