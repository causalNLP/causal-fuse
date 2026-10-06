from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = CODE_ROOT.parent
RESOURCE_ROOT = CODE_ROOT / "resources"
VENDOR_ROOT = CODE_ROOT / "vendor"

__all__ = ["CODE_ROOT", "RESOURCE_ROOT", "VENDOR_ROOT", "WORKSPACE_ROOT"]
