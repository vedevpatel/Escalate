"""
proxy_server.py

Boilerplate for intercepting and logging LLM API traffic via LiteLLM proxy.
"""

import os
import json
import logging

import litellm
from litellm.proxy.proxy_server import app, ProxyConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
