#!/usr/bin/env python
# coding: UTF-8

import orz
import os
import json


if __name__ == '__main__':

    print("test.json --> test.sta")
    orz.json2sta("test.json", "test.sta")
    print("test.sta --> test_out.json")
    if not os.path.isdir("in"):
        os.mkdir("in")
    orz.sta2json("test.sta", "in/")
    if not os.path.isdir("out"):
        os.mkdir("out")
    orz.json2sta("in/test.json", "out/")
    print(json.dumps(orz.sta2obj("out/test.sta"), indent=2))

    # orz.json2sta('workshop/VIPLFaceRecognizer4.3.SID.json', 'workshop/tmp')
    # orz.json2sta('workshop/VIPLFaceRecognizer5.0.RN2.json', 'workshop/tmp')
    #
    # orz.sta2json('workshop/tmp/VIPLFaceRecognizer5.0.RN2.sta')



