import pkgutil
import subprocess
import sys

import cvat_mcp.services

SERVICES = [m.name for m in pkgutil.iter_modules(cvat_mcp.services.__path__, "cvat_mcp.services.")]


def test_services_and_render_do_not_import_mcp():
    imports = "; ".join(f"import {name}" for name in [*SERVICES, "cvat_mcp.render"])
    code = f"import sys; {imports}; assert 'mcp' not in sys.modules, 'services must not use mcp'"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False, timeout=60
    )
    assert result.returncode == 0, result.stderr
