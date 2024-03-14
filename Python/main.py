import os
import pickle
from math import ceil
import sys


class Cache:
    def __init__(self, func):
        self.func = func
        self.cache = b""
        self.index = -1

    def __call__(self, instance, start, size):
        start = int(start)
        size = int(size)
        if (new_index := int(start / instance.buffer_size)) != self.index:
            max_buffers = ceil(instance.size / instance.buffer_size)
            instance._filestream.seek((new_index % max_buffers) * instance.buffer_size)
            self.cache = instance._filestream.read(instance.buffer_size)
            self.index = new_index
        elif (start + size) / instance.buffer_size > self.index + 1:
            instance._filestream.seek(start)
            return instance._filestream.read(size)
        start = start - (self.index * instance.buffer_size)
        return self.func(start, start + size, self.cache)


class BinaryFileHandler:
    @property
    def buffer_size(self):
        return self.__buffer_size

    @property
    def size(self):
        return os.path.getsize(self._filestream.name)

    def __init__(self, file_path, trunc=False, buffer_size=1024):
        mode = "wb+" if trunc else "rb+"
        self._filestream = open(os.path.abspath(file_path), mode)
        self.__buffer_size = buffer_size

    def count(self, *sequence):
        to_return = [0 for _ in range(len(sequence))]
        buffer_index = 0
        while buffer_index * self.buffer_size < self.size:
            buffer = self.read(buffer_index * self.buffer_size, self.buffer_size)
            last_ocurrence_index = 0
            for index, seq in enumerate(sequence):
                to_return[index] += buffer.count(seq)
                last_ocurrence_index = max(buffer.rfind(seq), last_ocurrence_index)
            if (last_ocurrence_index) == -1:
                buffer_index += 1
            else:
                buffer_index += (
                    ceil(last_ocurrence_index + len(sequence)) / self.buffer_size
                )

        return to_return

    @Cache
    def __read(start, size, caching):
        return caching[start:size]

    def read(self, start, size):
        return self.__read(self, start, size)

    def read_bin(self, start, size):
        _size = ceil(size / 8)

        return "".join(f"{byte:0>8b}" for byte in self.read(start // 8, _size))[
            start % 8 : (start % 8)+size
        ]

    def write(self, *data, start=None):
        if start != None:
            self._filestream.seek(start)
        for content in data:
            self._filestream.write(content)

    def write_hex(self, hex_string):
        self._filestream.write(bytes.fromhex(hex_string))


class HuffmanTreeNode:
    parent = None

    __childs = [None, None]

    @property
    def childs(self):
        return tuple(self.__childs)

    @property
    def is_leaf(self):
        return self.data != None

    @property
    def path(self):
        if self.parent is None:
            return ""
        else:
            return self.parent.path + str(self.parent.childs.index(self))

    @property
    def child_left(self):
        return self.childs[0]

    @property
    def child_right(self):
        return self.childs[1]

    def __init__(self, frequency=0, data=None):
        self.frequency = frequency
        self.data = data

    def set_childs(self, left_child, right_child):
        left_child.parent = self
        right_child.parent = self
        self.__childs = [left_child, right_child]

        self.frequency = left_child.frequency + right_child.frequency


class HuffmanFrequencies:
    def __init__(self, bin_filestream):
        self.__update(bin_filestream)

    def __update(self, bin_filestream):
        print(" Getting Frequencies...")
        sequence = [bytes.fromhex(f"{i:0>2x}") for i in range(256)]
        self.__frequency = sorted(
            [
                HuffmanTreeNode(frequency, data)
                for data, frequency in zip(sequence, bin_filestream.count(*sequence))
                if frequency > 0
            ],
            key=lambda item: item.frequency,
        )

    def get(self):
        return self.__frequency


class HuffmanCoding:
    def __get_bin_codes(self, frequency_queue):
        print(f" Obtaining binary codes...")
        leafs = tuple(frequency_queue)
        while len(frequency_queue) > 1:
            lower_frequencies = frequency_queue[:2]
            new_node = HuffmanTreeNode()
            new_node.set_childs(*lower_frequencies)
            del frequency_queue[:2]

            for index, item in enumerate(frequency_queue[::-1]):
                index = len(frequency_queue) - index - 1
                if item.frequency <= new_node.frequency:
                    frequency_queue.insert(index + 1, new_node)
                    break
                elif index == 0:
                    frequency_queue.insert(index, new_node)
        return {leaf.data: leaf.path for leaf in leafs}

    def __get_solver(self, bin_codes):
        to_return = {}
        current_dict = to_return

        for data, bin_code in bin_codes.items():
            for i, bit in enumerate(bin_code):
                if bit not in current_dict.keys():
                    current_dict[bit] = {"0": {}, "1": {}}
                if i == len(bin_code) - 1:
                    current_dict[bit] = data
                else:
                    current_dict = current_dict[bit]
            current_dict = to_return
        return to_return

    def encode(self, input_path, output_path):
        input_file = BinaryFileHandler(input_path)
        output_file = BinaryFileHandler(output_path, True)

        frequencies = HuffmanFrequencies(input_file)
        bin_codes = self.__get_bin_codes(frequencies.get())

        current_index = 0
        file_size = input_file.size
        to_write = []
        current_byte = []
        print(f" Encoding...")
        while current_index < file_size:
            current_byte.extend(bin_codes[input_file.read(current_index, 1)])
            current_index += 1
            if len(current_byte) >= 8:
                to_write.append(f"{int(''.join(current_byte[:8]), 2):0>2x}")
                del current_byte[:8]
            if len(to_write) >= output_file.buffer_size / 2:
                output_file.write_hex("".join(to_write))
                to_write.clear()
        if len(to_write) > 0:
            output_file.write_hex("".join(to_write))
        if len(current_byte) > 0:
            current_byte = f"{int(''.join(current_byte[:8]), 2):x}"
            output_file.write_hex(
                "".join(current_byte) + ("0" if len(current_byte) % 2 != 0 else "")
            )

        pickle.dump(
            self.__get_solver(bin_codes),
            open(output_path + ".key", "wb"),
        )

    def decode(self, file_path, key_path, output_path):
        print(f" Decoding...")
        input_file = BinaryFileHandler(file_path)
        output_file = BinaryFileHandler(output_path, True)
        key = pickle.load(open(key_path, "rb"))
        inner_key = key
        to_write = []
        current_index = 0
        file_size = input_file.size * 8
        while current_index < file_size:

            bit = input_file.read_bin(current_index, 1)
            inner_key = inner_key[bit]
            current_index += 1
            if type(inner_key) != dict:
                to_write.append(inner_key)
                inner_key = key
                if len(to_write) >= output_file.buffer_size:
                    output_file.write(b"".join(to_write))
                    to_write.clear()
        output_file.write(b"".join(to_write))


if __name__ == "__main__":
    try:
        mode = sys.argv[1]
        match mode:
            case "encode":
                input_path, output_path = sys.argv[2:]
                HuffmanCoding().encode(input_path, output_path)
            case "decode":
                input_path, key_path, output_path = sys.argv[2:]
                HuffmanCoding().decode(input_path, key_path, output_path)
    except ValueError as ex:
        print(ex)
    # HuffmanCoding().encode("test.mp4", "test")
