from rdkit import Chem
from rdkit.Chem import SDMolSupplier, SDWriter
import os
# 读取包含多个小分子的SDF文件
sdf_supplier = SDMolSupplier('D:/yanbujian/PyCharm Community Edition 2023.1/xiangmu1/xiangmu1.2/'
                             'force file data/tdcyp_data/cyp2d6_td_youhua.sdf')
active_directory = 'D:/yanbujian/PyCharm Community Edition 2023.1/xiangmu1/xiangmu1.2/' \
                   'force file data/tdcyp_data/cyp2d6_td/active/'
# cid_skip_file = "C:/Users/ididi/Desktop/skip_fenzi.txt"
inactive_directory = 'D:/yanbujian/PyCharm Community Edition 2023.1/xiangmu1/xiangmu1.2/' \
                     'force file data/tdcyp_data/cyp2d6_td/inactive/'
# i = '1'
# cids_to_skip = []
# # 读取要跳过的CID列表
# if os.path.exists(cid_skip_file):
#     with open(cid_skip_file, 'r') as file:
#         cids_to_skip = [line.strip() for line in file]
#         print(cids_to_skip)
for folder in [active_directory, inactive_directory]:
    if not os.path.exists(folder):
        os.makedirs(folder)
# 遍历每个小分子并将其保存为单独的SDF文件
for index, mol in enumerate(sdf_supplier):
    if mol is not None:
        # 从分子属性中获取CID
        nontoxic_value = mol.GetProp("Y")
        if nontoxic_value == '0':
            output_folder = inactive_directory
        if nontoxic_value == '1':
            output_folder = active_directory
        # 创建一个新的SDF文件写入器，以CID作为文件名
        output_filename = os.path.join(output_folder, f'{index}.sdf')
        writer = SDWriter(output_filename)
        writer.write(mol)
        writer.close()