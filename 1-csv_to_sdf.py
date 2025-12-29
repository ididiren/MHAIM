import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import SDWriter
from rdkit.Chem import PandasTools
import os
# 读取CSV文件

# 打印列名


df = pd.read_csv("D:/yanbujian/PyCharm Community Edition 2023.1/xiangmu1/xiangmu1.2/Ames/dataset_6/"
                 "ames_dataset6.csv")

sdf_path = 'D:/yanbujian/PyCharm Community Edition 2023.1/xiangmu1/xiangmu1.2/Ames/dataset_6/ames_dataset6.sdf'
# 根据SMILES列进行分组
grouped = df.groupby('smiles')
# 创建一个SDF文件来保存分子
writer = SDWriter(sdf_path)
# 遍历每个SMILES组
for smiles, group in grouped:
    # 从第一个分子中获取分子结构  # 去除引号
    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:
        # 添加分子属性
        for idx, row in group.iterrows():
            # mol.SetProp('CASRN', row['CASRN'])
            # mol.SetProp('DTXSID', row['DTXSID'])
            # mol.SetProp("_Name", row['Chemical_Name'])
            # mol.SetProp("Structure_Source", row["Structure_Source"])
            # mol.SetProp('SMILES', row["Canonical_QSARr"])
            # mol.SetProp("InChI_Code_QSARr", row["InChI_Code_QSARr"])
            # mol.SetProp("InChI_Key_QSARr", row["InChI_Key_QSARr"])
            # mol.SetProp("very_toxic", row["very_toxic"])
            # mol.SetProp("nontoxic", row["nontoxic"])
            mol.SetProp('labels', str(row['labels']).encode('utf-8').decode('utf-8'))
        # 生成初始3D构象
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)

        # 将分子写入SDF文件
        writer.write(mol)

writer.close()

print("SDF文件已生成")