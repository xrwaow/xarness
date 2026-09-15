"""Simple LZW compression/decompression (byte-oriented).

Usage:
    python compress.py compress  <in> <out>
    python compress.py decompress <in> <out>
"""
import struct
import sys


def compress(data: bytes) -> bytes:
    # Dictionary maps byte-strings to codes. Codes 0-255 are reserved for
    # single bytes; new entries start at 256.
    dictionary = {bytes([i]): i for i in range(256)}
    next_code = 256

    out = bytearray()

    def emit(code: int):
        # Codes are variable length: 2 bytes each (supports up to 65535 entries).
        out.extend(struct.pack(">H", code))

    w = b""
    for byte in data:
        c = bytes([byte])
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            emit(dictionary[w])
            dictionary[wc] = next_code
            next_code += 1
            if next_code > 65535:
                # Reset dictionary to keep codes within 2 bytes.
                dictionary = {bytes([i]): i for i in range(256)}
                next_code = 256
            w = c
    if w:
        emit(dictionary[w])
    return bytes(out)


def decompress(data: bytes) -> bytes:
    dictionary = {i: bytes([i]) for i in range(256)}
    next_code = 256

    out = bytearray()
    codes = struct.iter_unpack(">H", data)

    prev = None
    for (code,) in codes:
        if code in dictionary:
            entry = dictionary[code]
        elif code == next_code and prev is not None:
            entry = prev + prev[:1]
        else:
            raise ValueError(f"Invalid code {code} at position")

        out.extend(entry)
        if prev is not None:
            dictionary[next_code] = prev + entry[:1]
            next_code += 1
            if next_code > 65535:
                dictionary = {i: bytes([i]) for i in range(256)}
                next_code = 256
        prev = entry
    return bytes(out)


def main(argv):
    if len(argv) != 4:
        print(__doc__)
        sys.exit(1)
    mode, src, dst = argv[1], argv[2], argv[3]

    with open(src, "rb") as f:
        data = f.read()

    if mode == "compress":
        result = compress(data)
    elif mode == "decompress":
        result = decompress(data)
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)

    with open(dst, "wb") as f:
        f.write(result)

    print(f"{mode}: {src} ({len(data)} bytes) -> {dst} ({len(result)} bytes)")
    if mode == "compress":
        ratio = len(result) / len(data) * 100 if data else 0
        print(f"ratio: {ratio:.1f}% of original")


if __name__ == "__main__":
    main(sys.argv)
