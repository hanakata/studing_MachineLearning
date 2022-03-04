import lief

# バイナリのロード
binary = lief.parse("pe")
# ヘッダ内のMajorLinkerVersionの取得と出力
MajorLinkerVersion = binary.optional_header.major_linker_version
print(MajorLinkerVersion)

# ヘッダ内のMinorLinkerVersionの取得と出力
MinorLinkerVersion = binary.optional_header.minor_linker_version
print(MinorLinkerVersion)

# ヘッダ内のNumberOfSectionsの取得と出力
NumberOfSections = binary.header.numberof_sections
print(NumberOfSections)
# ヘッダ内のDebugSizeの取得と出力
DebugSize = binary.data_directories[6].size
print("DebugSize: {}".format(DebugSize))

# ヘッダ内のImageVersionの取得と出力
ImageVersion = binary.optional_header.major_image_version
print("ImageVersion: {}".format(ImageVersion))

# ヘッダ内のIatRVAの取得と出力
IATRVA = binary.data_directories[12].rva
print("IatRVA: {}".format(IATRVA))

# ヘッダ内のExportSizeの取得と出力
ExportSize = binary.data_directories[0].size
print("ExportSize: {}".format(ExportSize))

# ヘッダ内のResourceSizeの取得と出力
ResSize = binary.data_directories[2].size
print("ResourceSize: {}".format(ResSize))

# ヘッダ内のVirtualSize2の取得と出力
VirtualSize2 = binary.sections[1].virtual_size
print("VirtualSize2: {}".format(VirtualSize2))

# ヘッダ内のDebugRVAの取得と出力
NumberOfSections = binary.header.numberof_sections
print("NumberOfSections: {}".format(NumberOfSections))
import json

binary_json = lief.to_json(binary)
data = json.loads(binary_json)  
print(json.dumps(data, indent=4))