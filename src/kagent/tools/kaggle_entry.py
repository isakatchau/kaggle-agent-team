"""The official kaggle CLI, verifying TLS against the OS certificate store.

    python -m kagent kaggle auth login        # same as `kaggle auth login`

TLS-inspecting antivirus (Avast, Kaspersky...) and corporate proxies install their root
certificate in the OS store, which certifi, the bundle requests uses by default, never sees.
"""

import importlib.util
import sys
from pathlib import Path

import truststore

truststore.inject_into_ssl()
# kaggle only allows logged-out commands (`auth login`) when argv[0] is its own entry point
sys.argv[0] = str(Path(importlib.util.find_spec("kaggle").origin).with_name("__main__.py"))

from kaggle.cli import main  # noqa: E402  (must be imported after the injection)

if __name__ == "__main__":
    main()
