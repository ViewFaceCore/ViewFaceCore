#!/usr/bin/env python
# coding: UTF-8
"""
sta binary format:
[byte-layout:value] means an binary block with length and content
[[byte-layout:value], [byte-layout:value]] compose Piece
following are each type of pieces:
nil: [0:STA_NIL],[1:<hold>]
int: [0:STA_INT],[1-4:<int>]
float: [0:STA_INT],[1-4:<float>]
string: [0:STA_STRING],[1-4:<length>],[5-`4+<length>`:<string>]
binary: [0:STA_BINARY],[1-4:<length>],[5-`4+<length>`:<binary>]
list: [0:STA_LIST],[1-4:<length>],...[<value-any-piece>]x<length>
dict: [0:STA_DICT],[1-4:<length>],...[[<key-string-piece>],[<value-any-piece>]]x<length>
the top piece should be dict, so it can hold all contents.
"""

STA_NIL = 0
STA_INT = 1
STA_FLOAT = 2
STA_STRING = 3
STA_BINARY = 4
STA_LIST = 5
STA_DICT = 6

STA_MARK = 0x19910929

TS_SF_3_CODE = 0xAA
TS_SF_3_MAGIC = 0x20200202

class binary(object):
    """
    Enclosure for binary str object
    """
    def __init__(self, byte=''):
        """
        Init binary with content bytes
        :param byte: content
        """
        self.byte = byte

    def __str__(self):
        """
        Return the bytes of this binary
        :return: bytes
        """
        return str(self.byte)
