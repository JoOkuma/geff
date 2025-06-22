try:
    import rustworkx
except ImportError as err:
    raise ImportError(
        "The rustworkx submodule depends on rustworkx as an optional dependency. "
        "Please run `pip install geff[rustworkx]` to install the optional dependency."
    ) from err

from .io import read, write
