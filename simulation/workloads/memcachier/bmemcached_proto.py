# This file adapted from python-binary-memcached project of Jayson Reis.
# Original license below:
#
# (The MIT License)
# Copyright (c) 2011 Jayson Reis <santosdosreis@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


from datetime import datetime, timedelta
import logging
import re
import socket
import struct
import threading

logger = logging.getLogger(__name__)

OPCODE_TO_NAME = {
    0x00 : 	"Get",
    0x01 : 	"Set",
    0x02 : 	"Add",
    0x03 : 	"Replace",
    0x04 : 	"Delete",
    0x05 : 	"Increment",
    0x06 : 	"Decrement",
    0x07 : 	"Quit",
    0x08 : 	"Flush",
    0x09 : 	"GetQ",
    0x0a : 	"No-op",
    0x0b : 	"Version",
    0x0c : 	"GetK",
    0x0d : 	"GetKQ",
    0x0e : 	"Append",
    0x0f : 	"Prepend",
    0x10 : 	"Stat",
    0x11 : 	"SetQ",
    0x12 : 	"AddQ",
    0x13 : 	"ReplaceQ",
    0x14 : 	"DeleteQ",
    0x15 : 	"IncrementQ",
    0x16 : 	"DecrementQ",
    0x17 : 	"QuitQ",
    0x18 : 	"FlushQ",
    0x19 : 	"AppendQ",
    0x1a : 	"PrependQ",
    0x1b : 	"Verbosity",
    0x1c : 	"Touch",
    0x1d : 	"GAT",
    0x1e : 	"GATQ",
    0x20 : 	"SASL List mechs",
    0x21 : 	"SASL Auth",
    0x22 : 	"SASL Step",
    0x30 : 	"RGet",
    0x31 : 	"RSet",
    0x32 : 	"RSetQ",
    0x33 : 	"RAppend",
    0x34 : 	"RAppendQ",
    0x35 : 	"RPrepend",
    0x36 : 	"RPrependQ",
    0x37 : 	"RDelete",
    0x38 : 	"RDeleteQ",
    0x39 : 	"RIncr",
    0x3a : 	"RIncrQ",
    0x3b : 	"RDecr",
    0x3c : 	"RDecrQ",
    0x3d : 	"SetVBucket",
    0x3e : 	"GetVBucket",
    0x3f : 	"DelVBucket",
    0x40 : 	"TAPConnect",
    0x41 : 	"TAPMutation",
    0x42 : 	"TAPDelete",
    0x43 : 	"TAPFlush",
    0x44 : 	"TAPOpaque",
    0x45 : 	"TAPVBucketSet",
    0x46 : 	"TAPCheckpointStart",
    0x47 : 	"TAPCheckpointEnd"
}

class Header:
    def __init__(self, magic, opcode, **kwargs):
        self.magic = magic
        if self.magic == 0x80:
            self.msgtype = "Req"
        else:
            self.msgtype = "Res"
        self.opcode = opcode
        try:
            self.opname = OPCODE_TO_NAME[opcode]
        except KeyError:
            self.opname = "0x%x" % opcode

        self.__dict__.update(kwargs)

class Packet:
    def __init__(self, header, extras, key, value):
        self.header = header
        self.extras = extras
        self.key = key
        self.value = value

class Protocol(threading.local):
    """
    This class is used by Client class to communicate with server.
    """
    HEADER_STRUCT = '!BBHBBHLLQ'
    HEADER_SIZE = 24

    MAGIC = {
        'request': 0x80,
        'response': 0x81
    }


    # All structures will be appended to HEADER_STRUCT
    COMMANDS = {
        'get': {'command': 0x00, 'struct': '%ds'},
        'getk': {'command': 0x0C, 'struct': '%ds'},
        'getkq': {'command': 0x0D, 'struct': '%ds'},
        'set': {'command': 0x01, 'struct': 'LL%ds%ds'},
        'setq': {'command': 0x11, 'struct': 'LL%ds%ds'},
        'add': {'command': 0x02, 'struct': 'LL%ds%ds'},
        'addq': {'command': 0x12, 'struct': 'LL%ds%ds'},
        'replace': {'command': 0x03, 'struct': 'LL%ds%ds'},
        'delete': {'command': 0x04, 'struct': '%ds'},
        'incr': {'command': 0x05, 'struct': 'QQL%ds'},
        'decr': {'command': 0x06, 'struct': 'QQL%ds'},
        'flush': {'command': 0x08, 'struct': 'I'},
        'noop': {'command': 0x0a, 'struct': ''},
        'stat': {'command': 0x10},
        'auth_negotiation': {'command': 0x20},
        'auth_request': {'command': 0x21, 'struct': '%ds%ds'},
    }

    STATUS = {
        'success': 0x00,
        'key_not_found': 0x01,
        'key_exists': 0x02,
        'auth_error': 0x08,
        'unknown_command': 0x81,

        # This is used internally, and is never returned by the server.  (The server returns a 16-bit
        # value, so it's not capable of returning this value.)
        'server_disconnected': 0xFFFFFFFF,
    }

    FLAGS = {
        'pickle': 1 << 0,
        'integer': 1 << 1,
        'long': 1 << 2,
        'compressed': 1 << 3
    }

    MAXIMUM_EXPIRE_TIME = 0xfffffffe

    COMPRESSION_THRESHOLD = 128

    @classmethod
    def read_header(self, header):
        (magic, opcode, keylen, extlen, datatype, status, bodylen, opaque,
         cas) = struct.unpack(Protocol.HEADER_STRUCT, header)
        header = Header(magic, opcode, keylen=keylen, extlen=extlen, 
                        datatype = datatype,
                        status = status, bodylen = bodylen, opaque = opaque, 
                        cas = cas)
        return header

    @classmethod
    def read_packet(self, packet_binary, max_len):
        if max_len < 24:
            raise IndexError("Max Len < Header")
        header_bytestr = packet_binary[:24]
        header = Protocol.read_header(header_bytestr)

        body_len = min(max_len - 24, header.bodylen)
        body = packet_binary[24:(24+body_len)]

        extras, key, value = '','',''
        if body_len > 0:
            if header.extlen > 0:
                extras = body[0:header.extlen]
            if header.keylen > 0:
                key = body[header.extlen : header.keylen]
            if header.bodylen > (header.extlen + header.keylen):
                value = body[(header.extlen + header.keylen):]
        
        return Packet(header, extras, key, value)
            
