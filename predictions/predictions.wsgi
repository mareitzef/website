#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os

# Add the application directory to the sys.path
app_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, app_dir)

from app import app as application

if __name__ == "__main__":
    application.run()
