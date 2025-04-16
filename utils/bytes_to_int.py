
import struct

filename = 'your_filename_path'
file_path = 'your_filepath'
file_path += filename

bytes = []
with open(file_path, "rb") as file:
    while True:
        byte = file.read(2)
        if not byte:
            break

        number = int.from_bytes(byte, byteorder='little')
        signed_number = struct.unpack('<h', byte)[0]  # data are little endian and short int
        bytes.append(signed_number)

final_str = ",".join(map(str,bytes))
with open(file_path+"out", "w") as file:
    file.write(final_str)
print(f"{filename} - Number of parameters = ", len(bytes))
