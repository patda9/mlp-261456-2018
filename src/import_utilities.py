import json, os, sys

_abs = os.path.abspath
_dirname = os.path.dirname
_join = os.path.join

class ImportUtilities:
    def __init__(self):
        self.paths = json.load(open(_abs(_join(_dirname(__file__), "../../paths.json"))))["paths"]

        self.share_module_paths(self.paths)

    def share_module_paths(self, modules_paths):
        for path in self.paths:
            sys.path.insert(1, _abs(_join(_dirname(__file__), "../..", path)))
