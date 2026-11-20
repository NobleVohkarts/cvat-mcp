import subprocess
import sys

CODE = """
import sys
import cvat_mcp.render
import cvat_mcp.services.annotations
import cvat_mcp.services.issues
import cvat_mcp.services.server_info
import cvat_mcp.services.tasks
assert "mcp" not in sys.modules, "services must not depend on mcp"
"""


def test_services_and_render_do_not_import_mcp():
    result = subprocess.run(
        [sys.executable, "-c", CODE], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
