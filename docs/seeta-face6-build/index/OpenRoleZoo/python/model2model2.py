#!/usr/bin/env python

try:
    import orz
except ImportError:
    import os
    import sys
    curr_path = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(os.path.join(curr_path, "D:\Working\OpenRoleZoo\OpenRoleZoo\python"))
    import orz

import os

def timestamp(filename):
    if not os.path.exists(filename):
        return 0
    statinfo = os.stat(filename)
    return statinfo.st_mtime

if __name__ == '__main__':
    input_dir = 'rawmd'
    output_dir = 'stamd'

    filenames = os.listdir(input_dir)

    count_keep = 0
    count_modify = 0

    for filename in filenames:
        if filename[-5:] != '.json':
            continue
        name, ext = os.path.splitext(filename)
        filename_sta = name + '.sta'
        input_filename = os.path.join(input_dir, filename)
        output_filename = os.path.join(output_dir, filename_sta)
        if timestamp(output_filename) < timestamp(input_filename):
            print('Converting %s' % input_filename)
            orz.json2sta(input_filename, output_filename)
            count_modify += 1
        else:
            print('Keeping %s' % input_filename)
            count_keep += 1

    count_total = count_modify + count_keep

    print("Total: %d. Modified: %d, kept: %d" % (count_total, count_modify, count_keep))