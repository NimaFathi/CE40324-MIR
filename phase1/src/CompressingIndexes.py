from src.Compressor import Compressor
from src.PositionalIndex import PositionalIndex


# this file is to help compress and decompress using the two type pf compressions defined in the cpmpressor,
# the gamma type plus the variable byte encoding type

def compress(compression_type='var_byte'):
    compressed_index = dict()
    for term in PositionalIndex.index.keys():
        compressed_index[term] = dict()
        for doc_id in PositionalIndex.index[term]:

            if compression_type == 'gamma':
                compressed_index[term][doc_id] = Compressor.gamma_encode(PositionalIndex.index[term][doc_id])
            elif compression_type == 'var_byte':
                compressed_index[term][doc_id] = Compressor.variable_byte_encode(PositionalIndex.index[term][doc_id])

    return compressed_index


def decompress(compresses_index, compression_type='var_byte'):
    PositionalIndex.index = dict()
    for term in compresses_index.keys():
        PositionalIndex.index[term] = dict()
        for doc_id in compresses_index[term]:

            if compression_type == 'gamma':
                PositionalIndex.index[term][doc_id] = Compressor.gamma_decode(compresses_index[term][doc_id])
            if compression_type == 'var_byte':
                PositionalIndex.index[term][doc_id] = Compressor.variable_byte_decode(compresses_index[term][doc_id])
