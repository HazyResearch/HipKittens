import os

import lit.formats
from lit.llvm import llvm_config

config.name = "HK"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".mlir"]
config.excludes = ["Inputs", "CMakeLists.txt", "lit.cfg.py", "lit.site.cfg.py"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.hk_obj_root, "test")

llvm_config.use_default_substitutions()
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.add_tool_substitutions(
    ["hk-opt", "hk-translate"], [config.hk_tools_dir, config.llvm_tools_dir])
