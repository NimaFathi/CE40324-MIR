from __future__ import division

import pickle
from math import floor, log2

from struct import pack, unpack


class Compressor:

    @staticmethod
    def calculate_gaps(numbers):
        if len(numbers) == 0:
            return []
        gaps = [numbers[0]]
        for i in range(len(numbers) - 1):
            gaps.append(numbers[i + 1] - numbers[i])
        return gaps

    @staticmethod
    def return_list(gaps):
        if len(gaps) == 0:
            return []
        numbers = [gaps[0]]
        for i in range(len(gaps) - 1):
            numbers.append(numbers[i] + gaps[i + 1])
        return numbers

    @staticmethod
    def variable_byte_encode(numbers):

        gaps = Compressor.calculate_gaps(numbers)
        res = ""
        for n in gaps:
            res += Compressor.variable_byte_encode_number(n)
        return Compressor.bin_to_byte(res)

    @staticmethod
    def variable_byte_encode_number(number):

        # bytes_list = []
        # while True:
        #     bytes_list.insert(0, number % 128)
        #     if number < 128:
        #         break
        #     number = number // 128
        # bytes_list[-1] += 128
        # return pack('%dB' % len(bytes_list), *bytes_list)
        s = ""
        bytes_list = []
        while True:
            binary_num = bin(number % 128)[2:]
            bytes_list.append('0' * (8 - len(binary_num)) + str(binary_num))
            if number < 128:
                break
            number = number // 128
        low_byte = list(bytes_list[0])
        low_byte[0] = '1'
        bytes_list[0] = "".join(low_byte)

        for i in range(len(bytes_list) - 1, -1, -1):
            s += bytes_list[i]

        return s

    @staticmethod
    def var_answer(self, bytes_list, s):
        for i in range(len(bytes_list) - 1, -1, -1):
            s += bytes_list[i]

    @staticmethod
    def variable_byte_decode(bytestream):

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
        return Compressor.return_list(numbers)

    @staticmethod
    def gamma_encode(numbers):
        gaps = Compressor.calculate_gaps(numbers)
        code_str = ""
        for n in gaps:
            code_str += Compressor.gamma_encode_number(n)
        return Compressor.bin_to_byte(code_str)

    @staticmethod
    def gamma_encode_number(number):
        code = ""
        for _ in range(floor(log2(number))):
            code += '1'
        binary_num = bin(number)[2:]
        code += "0"
        for i in range(1, len(binary_num)):
            code += binary_num[i]
        return code

    @staticmethod
    def gamma_decode(codes):

        converted_str = Compressor.byte_to_bin(codes)
        l = len(converted_str)
        pos = 0
        gaps = []
        while pos < l:
            counter = 0
            while converted_str[pos + counter] == '1':
                counter += 1
            n = 1
            for i in range(pos + counter + 1, pos + 2 * counter + 1):
                n *= 2
                if converted_str[i] == '1':
                    n += 1
            pos += 2 * counter + 1
            gaps.append(n)
        return Compressor.return_list(gaps)

    @staticmethod
    def bin_to_byte(bin_str):
        num = int(bin_str, 2)
        b = bytearray()
        while num:
            b.append(num & 0xff)
            num >>= 8
        return bytes(b[::-1])

    @staticmethod
    def byte_to_bin(byte_arr):
        str = ""
        f_1 = True

        for b in byte_arr:
            byte_str = str(bin(b)[2:])
            if not f_1:
                l = len(byte_str)
                str += '0' * (8 - l)
            else:
                f_1 = False
            str += byte_str
        return str
