from __future__ import division
from struct import pack, unpack


class Compressor:
    @staticmethod
    def convert_binary_str_to_bytes(bin_str):
        n = int(bin_str, 2)
        b = bytearray()
        while n:
            b.append(n & 0xff)
            n >>= 8
        return bytes(b[::-1])

    @staticmethod
    def convert_bytes_to_binary_str(byte_arr):
        result = ""
        first_byte_flag = True
        for b in byte_arr:
            byte_str = str(bin(b)[2:])
            if not first_byte_flag:
                result += '0' * (8 - len(byte_str))
            else:
                first_byte_flag = False
            result += byte_str
        return result


def var_encode_number(number):
    bytes_list = []
    while True:
        bytes_list.insert(0, number % 128)
        if number < 128:
            break
        number = number // 128
    bytes_list[-1] += 128
    return pack('%dB' % len(bytes_list), *bytes_list)


def var_encode(numbers):
    bytes_list = []
    for number in numbers:
        bytes_list.append(var_encode_number(number))
    return b"".join(bytes_list)


def decode(bytestream):
    n = 0
    numbers = []
    bytestream = unpack('%dB' % len(bytestream), bytestream)
    for byte in bytestream:
        if byte < 128:
            n = 128 * n + byte
        else:
            n = 128 * n + (byte - 128)
            numbers.append(n)
            n = 0
    return numbers
