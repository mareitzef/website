#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# Path to project directory
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

# Activate venv
activate_this = "/home/zef/html/predictions/venv/bin/activate_this.py"
exec(open(activate_this).read(), {"__file__": activate_this})

# Import Flask app
from app import app as application
