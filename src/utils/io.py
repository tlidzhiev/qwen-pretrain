from pathlib import Path


def get_root() -> Path:
    """
    Get the root directory of the project.

    Returns
    -------
    Path
        Root directory path.
    """
    ROOT = Path(__file__).absolute().resolve().parent.parent.parent
    return ROOT
