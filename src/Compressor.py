
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
