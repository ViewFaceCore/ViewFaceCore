#!/usr/bin/env python
# coding: UTF-8

import orz
import json
import sys


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage viewsta.py filename.sta")
        exit()

    print(json.dumps(orz.sta2obj(sys.argv[1], binary_mode=2), indent=2))