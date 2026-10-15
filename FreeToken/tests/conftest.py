"""
Author: FreeTokenBuild
Date:   2026
Email:  eyeclept@pm.me

Description:
    Put FreeToken/ on sys.path so tests can import broker_client and
    serve_wrapper without installing a package.
"""
# Imports
import logging
import sys
from pathlib import Path

# Globals
logger = logging.getLogger(__name__)

_FREE_TOKEN_DIR = Path(__file__).resolve().parents[1]
if str(_FREE_TOKEN_DIR) not in sys.path:
    sys.path.insert(0, str(_FREE_TOKEN_DIR))
