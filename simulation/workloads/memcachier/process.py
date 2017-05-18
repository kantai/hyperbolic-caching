#
#
# USAGE: 
#
#
#


import binascii
import sys

from bmemcached_proto import Protocol as MCProtocol

client_name_cache = {}

def handle_packet(str_packet, just_headers):
    try:
        date, size, hexdata, port = str_packet.split('\t')
    except ValueError:
        sys.stderr.write(str_packet + "\n")
        return ""

    try:
        bytestr = binascii.unhexlify(hexdata)
    except TypeError:
        return ""

    req_find = bytestr.find("\x80")
    res_find = bytestr.find("\x81")
    
    if req_find == -1:
        find = res_find
    elif res_find == -1:
        find = req_find
    else:
        find = min(req_find, res_find)

#    if find == res_find:
#        return ""

    if find == req_find and find != 15:
        return "ErrorReq : req_off : %s :: %i" % (hexdata, find)
    if find == res_find and find != 0:
        return "ErrorRes : req_off : %s :: %i" % (hexdata, find)

    if just_headers:
        if find == res_find:
            return ""
        client_len = ord(bytestr[0])
        client_name = bytestr[1:1+client_len]
        size = bytestr[1+client_len:find]
        if client_name in client_name_cache:
            return ""
        else:
            client_name_cache[client_name] = True
            if not all(ord(c) < 128 for c in client_name):
                return ""
            return " %s --> %s " % (client_name,
                                    int("%s" % binascii.hexlify(size), 16))

    if find == req_find:
        port = port.split(',')[1]
    else:
        port = port.split(',')[0]
    port = int(port)

    if find > 0:
#        tcp_heading = hexdata[8*2:(find*2)]
        client_len = ord(bytestr[0])
        client_name = bytestr[1:1+client_len]
        tcp_heading = client_name
    else:
        tcp_heading = ''

    bytestr = bytestr[find:]

    max_len = int(size) - find

    try:
        packet = MCProtocol.read_packet(bytestr, max_len)
    except IndexError:
        return "ErrorHeader : maxlen : %s :: %i" % (hexdata, find)

    value = packet.value
    if len(value) > 10:
        value = value[:10]

    val_len = packet.header.bodylen - packet.header.keylen - packet.header.extlen

    output_line = "'%s'\t%s\t%s\t'%s'\t%d\t'%s'\t'%s'\t'%s'\t%d" % (
        date,
        packet.header.msgtype, packet.header.opname, 
        packet.key, val_len, 
        binascii.hexlify(value), 
        tcp_heading, packet.header.opaque, port)
    return output_line

def main(input, just_headers = False):
    for ix, str_packet in enumerate(input):
        if ix % 10000 == 0:
            sys.stderr.write("\rprocessed %iK" % (ix / 1000))
            sys.stderr.flush()

        str_packet 
        packet_str = handle_packet(str_packet, just_headers)
        if packet_str != "":
            if not packet_str.startswith("Error"):
                print packet_str
    sys.stderr.write("\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input = open(sys.argv[1], 'r')
        if len(sys.argv) > 2:
            main(input, just_headers = sys.argv[2] == '--just-headers')
    else:
        input = sys.stdin
    main(input)
    input.close()
