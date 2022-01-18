#!/usr/bin/env python
# coding: UTF-8

import os
import struct
from .sta import *
import json
import copy
import base64
from collections import OrderedDict


class Stream:
    def __init__(self, byte):
        self.byte = byte
        self.index = 0

    def read(self, size=None):
        data = ''
        if size is None:
            data = self.byte[self.index:]
        else:
            data = self.byte[self.index:self.index+size]
        self.index += len(data)
        return data


def unpack_nil(stream, **kwargs):
    stream.read(1)
    return None


def unpack_int(stream, **kwargs):
    return struct.unpack('=i', stream.read(4))[0]


def unpack_float(stream, **kwargs):
    return struct.unpack('=f', stream.read(4))[0]


def unpack_string(stream, **kwargs):
    length = struct.unpack('=i', stream.read(4))[0]
    s = struct.unpack('=%ds' % length, stream.read(length))[0].decode()
    return s


def unpack_binary(stream, **kwargs):
    length = struct.unpack('=i', stream.read(4))[0]
    s = struct.unpack('=%ds' % length, stream.read(length))[0]

    mode = 0
    if 'binary_mode' in kwargs:
        mode = kwargs['binary_mode']

    if mode == 0:
        return '@base64@%s' % base64.b64encode(s)
    elif mode == 1:
        # save file
        if 'getway' not in kwargs:
            raise Exception("getway must be set.")
        if 'workshop' not in kwargs:
            raise Exception("workshop must be set.")
        filename_ext = kwargs['getway'] + '.bin'
        binary_filename = os.path.join(kwargs['workshop'], filename_ext)

        s[8] = 1
        with open(binary_filename, 'wb') as f:
            f.write(s)
        return '@file@%s' % filename_ext
    elif mode == 2:
        return '@binary@%d' % length
    else:
        return binary(s)


def unpack_list(stream, **kwargs):
    local_kwargs = copy.copy(kwargs)
    if 'getway' not in local_kwargs:
        local_kwargs['getway'] = ''
    getway = local_kwargs['getway']

    obj = []
    length = struct.unpack('=i', stream.read(4))[0]
    for i in range(length):
        local_kwargs['getway'] = getway + '_' + str(i)
        obj.append(unpack_obj(stream, **local_kwargs))
    return obj


def unpack_dict(stream, **kwargs):
    local_kwargs = copy.copy(kwargs)
    if 'getway' not in local_kwargs:
        local_kwargs['getway'] = ''
    getway = local_kwargs['getway']

    obj = {}
    length = struct.unpack('=i', stream.read(4))[0]
    for i in range(length):
        key = unpack_string(stream, **kwargs)
        local_kwargs['getway'] = getway + '_' + key
        value = unpack_obj(stream, **local_kwargs)
        obj[key] = value
    obj = OrderedDict(sorted(obj.items(), key=lambda item: item[0]))
    return obj


def unpack_obj(stream, **kwargs):
    """
    Convert an stream(sta format) to object(json format)
    :param stream: Stream of binary sta file
    :param workshop: path to write binary file
    :param getway: the getway to all values
    :param binary_mode: 0(default): means write @base64@...
                        1: means @file@path
                        2: means write @binary@size
                        3: means str for binary memory
    :return: unpacked object
    """
    mark = struct.unpack('=b', stream.read(1))[0]
    if mark == STA_NIL:
        return unpack_nil(stream, **kwargs)
    elif mark == STA_INT:
        return unpack_int(stream, **kwargs)
    elif mark == STA_FLOAT:
        return unpack_float(stream, **kwargs)
    elif mark == STA_STRING:
        return unpack_string(stream, **kwargs)
    elif mark == STA_BINARY:
        return unpack_binary(stream, **kwargs)
    elif mark == STA_LIST:
        return unpack_list(stream, **kwargs)
    elif mark == STA_DICT:
        return unpack_dict(stream, **kwargs)
    else:
        raise Exception("Unsupported mark type: ", type(mark))


def sta2obj(sta_filename, **kwargs):
    """
    Convert filename.sta to object
    :param sta_filename: input sta filename
    :param binary_mode: 0(default): means write @base64@...
                        1: means @file@path
                        2: means write @binary@size
                        3: means str for binary memory
    :return:
    """

    byte = ''
    with open(sta_filename, 'rb') as ifile:
        byte = ifile.read()

    stream = Stream(byte)

    mark = struct.unpack('=i', stream.read(4))[0]

    if mark != STA_MARK:
        raise Exception("%s is not a valid sta file." % sta_filename)

    # kwargs = {}
    if 'binary_mode' not in kwargs:
        kwargs['binary_mode'] = 0

    obj = unpack_obj(stream, **kwargs)

    return obj


def sta2json(sta_filename, json_filename=None, **kwargs):
    """
    Convert filename.sta to filename.json.
    :param sta_filename: input sta filename
    :param json_filename: output json filename or path
    :param binary_mode: 0(default): means write @base64@...
                        1: means @file@path
                        2: means write @binary@size
                        3: means str for binary memory
    :return:
    """

    filepath, filename_ext = os.path.split(sta_filename)
    filename, ext = os.path.splitext(filename_ext)

    if json_filename is None:
        json_filename = os.path.join(filepath, filename + ".json")

    if os.path.isdir(json_filename):
        json_filename = os.path.join(json_filename, filename + ".json")

    workshop, getway_ext = os.path.split(json_filename)
    getway = os.path.splitext(getway_ext)[0]

    if len(workshop) > 0 and not os.path.isdir(workshop):
        raise Exception("%s/ is not a valid path." % workshop)

    with open(json_filename, 'w') as ofile:
        byte = ''
        with open(sta_filename, 'rb') as ifile:
            byte = ifile.read()

        stream = Stream(byte)

        mark = struct.unpack('=i', stream.read(4))[0]

        if mark != STA_MARK:
            raise Exception("%s is not a valid sta file." % sta_filename)

        kwargs['workshop'] = workshop
        kwargs['getway'] = getway
        if 'binary_mode' not in kwargs:
            kwargs['binary_mode'] = 1

        obj = unpack_obj(stream, **kwargs)

        json.dump(obj, ofile, indent=2)
