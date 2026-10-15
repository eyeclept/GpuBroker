"""
Author: Richard Baldwin
Date:   2026
Email:  eyeclept@pm.me

Description:
    Put TI-Template/ on sys.path so tests can import
    async_broker_client and sync_broker_client.
"""
# Imports
import logging
import sys
from pathlib import Path

# Globals
logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).resolve().parents[1]
if str(_TEMPLATE_DIR) not in sys.path:
    sys.path.insert(0, str(_TEMPLATE_DIR))
