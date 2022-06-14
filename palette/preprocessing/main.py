from huffman import HuffmanCoding


def main():
    # Start reading
    paths = []

    for path in paths:
        h = HuffmanCoding(path)
        output_path = h.compress()
        h.decompress(output_path)
