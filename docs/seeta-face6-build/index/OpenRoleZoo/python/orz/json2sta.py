#!/usr/bin/env python
# coding: utf-8

import json
import os
from .sta import *
import struct
import time
import base64


_registered_command = {}


def register_command(cmd, callback):
    _registered_command[cmd] = callback


def pack_command(cmd, **kwargs):
    if len(cmd) == 0 or cmd[0] != '@':
        return pack_string(cmd)
    key_args = cmd[1:].split('@')
    key = key_args[0]
    args = key_args[1:]
    if key in _registered_command:
        return _registered_command[key](*args, **kwargs)
    return pack_string(cmd, **kwargs)


def pack_nil(**kwargs):
    byte = struct.pack('=bb', STA_NIL, 0)
    return byte


def pack_int(obj, **kwargs):
    byte = struct.pack('=bi', STA_INT, int(obj))
    return byte


def pack_float(obj, **kwargs):
    byte = struct.pack('=bf', STA_FLOAT, float(obj))
    return byte


def pack_raw_string(obj, **kwargs):
    s = str(obj)
    if not isinstance(s, bytes):
        s = s.encode()
    byte = struct.pack('=i%ds' % len(s), len(s), s)
    return byte


def pack_string(obj, **kwargs):
    s = str(obj)
    if not isinstance(s, bytes):
        s = s.encode()
    byte = struct.pack('=bi%ds' % len(s), STA_STRING, len(s), s)
    return byte


def pack_binary(obj, **kwargs):
    s = obj
    if not isinstance(s, bytes):
        s = s.encode()
    byte = struct.pack('=bi%ds' % len(s), STA_BINARY, len(s), s)
    return byte


def pack_list(obj, **kwargs):
    byte = struct.pack('=bi', STA_LIST, len(obj))
    for value in obj:
        byte += pack_obj(value, **kwargs)
    return byte


def pack_dict(obj, **kwargs):
    byte = struct.pack('=bi', STA_DICT, len(obj))
    keys = sorted(obj.keys())
    for key in keys:
        value = obj[key]
        byte += pack_raw_string(key, **kwargs)
        byte += pack_obj(value, **kwargs)
    return byte


def pack_value(value, **kwargs):
    try:
        if isinstance(value, unicode):
            return pack_value(value.encode("UTF-8"), **kwargs)
    except NameError:
        pass

    if value is None:
        return pack_nil(**kwargs)
    elif isinstance(value, str):
        if len(value) > 0 and value[0] == '@':
            return pack_command(value, **kwargs)
        return pack_string(value, **kwargs)
    elif isinstance(value, int):
        return pack_int(value, **kwargs)
    elif isinstance(value, float):
        return pack_float(value, **kwargs)
    elif isinstance(value, binary):
        return pack_binary(value, **kwargs)
    else:
        raise Exception("Unsupported value type: ", type(value))


def pack_obj(obj, **kwargs):
    """
    Convert object(json format) to byte array(sta format)
    :param obj: ready to convert
    :param workshop: means path to read binary file
    :return: str including byte array
    """

    if isinstance(obj, dict):
        return pack_dict(obj, **kwargs)
    elif isinstance(obj, list):
        return pack_list(obj, **kwargs)
    else:
        return pack_value(obj, **kwargs)


def pack_date(**kwargs):
    date = time.strftime("%Y-%m-%d", time.localtime())
    return pack_string(date, **kwargs)


def pack_time(**kwargs):
    date = time.strftime("%H:%M:%S", time.localtime())
    return pack_string(date, **kwargs)


def pack_datetime(**kwargs):
    date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return pack_string(date, **kwargs)


def pack_file(filename, **kwargs):
    if 'workshop' in kwargs:
        filename = os.path.join(kwargs['workshop'], filename)
    if not os.path.isfile(filename):
        raise Exception("%s is not a valid file." % filename)

    data = ''
    with open(filename, 'rb') as f:
        data = f.read()

    if not isinstance(data, bytes):
        data = data.encode()
    #data[8] = TS_SF_3_CODE 
    magic = TS_SF_3_MAGIC
    d = struct.pack('=Bi', TS_SF_3_CODE, magic)
    data = data[0:8] + d + data[13:] 
    return pack_binary(data, **kwargs)


def pack_base64(s, **kwargs):
    return pack_binary(base64.b64decode(s), **kwargs)


def pack_error(**kwargs):
    raise NotImplementedError


register_command("date", pack_date)
register_command("time", pack_time)
register_command("datetime", pack_datetime)
register_command("file", pack_file)
register_command("base64", pack_base64)
register_command("binary", pack_error)
register_command("nil", pack_nil)


def obj2sta(obj, sta_filename, workshop=None):
    """
    Convert an object(json format) to sta file
    :param obj: ready to convert
    :param sta_filename: output sta filename
    :param workshop: means path to read binary file
    :return:
    """

    if workshop is None:
        workshop = ''

    with open(sta_filename, 'wb') as ofile:

        kwargs = {}
        kwargs['workshop'] = workshop

        byte = pack_obj(obj, **kwargs)

        # write header
        ofile.write(struct.pack('i', STA_MARK))
        # write content
        ofile.write(byte)


def json2sta(json_filename, sta_filename=None):
    """
    Convert json file to sta file
    :param json_filename: input json filename
    :param sta_filename: output sta filename or path
    :return:
    """

    filepath, filename_ext = os.path.split(json_filename)
    filename, ext = os.path.splitext(filename_ext)

    if sta_filename is None:
        sta_filename = os.path.join(filepath, filename + ".sta")

    if os.path.isdir(sta_filename):
        sta_filename = os.path.join(sta_filename, filename + ".sta")

    output_path = os.path.split(sta_filename)[0]
    workshop = filepath

    if len(output_path) > 0 and not os.path.isdir(output_path):
        raise Exception("%s/ is not a valid path." % output_path)

    with open(sta_filename, 'wb') as ofile:
        obj = {}
        with open(json_filename) as ifile:
            obj = json.load(ifile)

        kwargs = {}
        kwargs['workshop'] = workshop

        byte = pack_obj(obj, **kwargs)

        # write header
        ofile.write(struct.pack('=i', STA_MARK))
        # write content
        ofile.write(byte)
