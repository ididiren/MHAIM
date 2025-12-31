import glob
import torch
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, ChemicalForceFields, SDMolSupplier
import random
import json
from atom_type_xindingyi import *
import warnings

warnings.filterwarnings('ignore', message='Molecule does not have explicit Hs. Consider calling AddHs()')
torch.cuda.manual_seed(123)
torch.manual_seed(123)
np.random.seed(123)
random.seed(123)
# MMFF94不同原子类型对应的力场参数
ff_param = [
    [1.050, 2.490, 3.890, 1.282],
    [1.350, 2.490, 3.890, 1.282],
    [1.100, 2.490, 3.890, 1.282],
    [1.300, 2.490, 3.890, 1.282],
    [0.250, 0.800, 4.200, 1.209],
    [0.700, 3.150, 3.890, 1.282],
    [0.650, 3.150, 3.890, 1.282],
    [1.150, 2.820, 3.890, 1.282],
    [0.900, 2.820, 3.890, 1.282],
    [1.000, 2.820, 3.890, 1.282],
    [0.350, 3.480, 3.890, 1.282],
    [2.300, 5.100, 3.320, 1.345],
    [3.400, 6.000, 3.190, 1.359],
    [5.500, 6.950, 3.080, 1.404],
    [3.000, 4.800, 3.320, 1.345],
    [3.900, 4.800, 3.320, 1.345],
    [2.700, 4.800, 3.320, 1.345],
    [2.100, 4.800, 3.320, 1.345],
    [4.500, 4.200, 3.320, 1.345],
    [1.050, 2.490, 3.890, 1.282],
    [0.150, 0.800, 4.200, 1.209],
    [1.100, 2.490, 3.890, 1.282],
    [0.150, 0.800, 4.200, 1.209],
    [0.150, 0.800, 4.200, 1.209],
    [1.600, 4.500, 3.320, 1.345],
    [3.600, 4.500, 3.320, 1.345],
    [0.150, 0.800, 4.200, 1.209],
    [0.150, 0.800, 4.200, 1.209],
    [0.150, 0.800, 4.200, 1.209],
    [1.350, 2.490, 3.890, 1.282],
    [0.150, 0.800, 4.200, 1.209],
    [0.750, 3.150, 3.890, 1.282],
    [0.150, 0.800, 4.200, 1.209],
    [1.000, 2.820, 3.890, 1.282],
    [1.500, 3.150, 3.890, 1.282],
    [0.150, 0.800, 4.200, 1.209],
    [1.350, 2.490, 3.890, 1.282],
    [0.850, 2.820, 3.890, 1.282],
    [1.100, 2.820, 3.890, 1.282],
    [1.000, 2.820, 3.890, 1.282],
    [1.100, 2.490, 3.890, 1.282],
    [1.000, 2.820, 3.890, 1.282],
    [1.000, 2.820, 3.890, 1.282],
    [3.000, 4.800, 3.320, 1.345],
    [1.150, 2.820, 3.890, 1.282],
    [1.300, 2.820, 3.890, 1.282],
    [1.000, 2.820, 3.890, 1.282],
    [1.200, 2.820, 3.890, 1.282],
    [1.000, 3.150, 3.890, 1.282],
    [0.150, 0.800, 4.200, 1.209],
    [0.400, 3.150, 3.890, 1.282],
    [0.150, 0.800, 4.200, 1.209],
    [1.000, 2.820, 3.890, 1.282],
    [1.300, 2.820, 3.890, 1.282],
    [0.800, 2.820, 3.890, 1.282],
    [0.800, 2.820, 3.890, 1.282],
    [1.000, 2.490, 3.890, 1.282],
    [0.800, 2.820, 3.890, 1.282],
    [0.650, 3.150, 3.890, 1.282],
    [1.800, 2.490, 3.890, 1.282],
    [0.800, 2.820, 3.890, 1.282],
    [1.300, 2.820, 3.890, 1.282],
    [1.350, 2.490, 3.890, 1.282],
    [1.350, 2.490, 3.890, 1.282],
    [1.000, 2.820, 3.890, 1.282],
    [0.750, 2.820, 3.890, 1.282],
    [0.950, 2.820, 3.890, 1.282],
    [0.900, 2.820, 3.890, 1.282],
    [0.950, 2.820, 3.890, 1.282],
    [0.870, 3.150, 3.890, 1.282],
    [0.150, 0.800, 4.200, 1.209],
    [4.000, 4.800, 3.320, 1.345],
    [3.000, 4.800, 3.320, 1.345],
    [3.000, 4.800, 3.320, 1.345],
    [4.000, 4.500, 3.320, 1.345],
    [1.200, 2.820, 3.890, 1.282],
    [1.500, 5.100, 3.320, 1.345],
    [1.350, 2.490, 3.890, 1.282],
    [1.000, 2.820, 3.890, 1.282],
    [1.000, 2.490, 3.890, 1.282],
    [0.800, 2.820, 3.890, 1.282],
    [0.950, 2.820, 3.890, 1.282],
    [0.450, 6.000, 4.000, 1.400],
    [0.550, 6.000, 4.000, 1.400],
    [1.400, 3.480, 3.890, 1.282],
    [4.500, 5.100, 3.320, 1.345],
    [6.000, 6.000, 3.190, 1.359],
    [0.150, 2.000, 4.000, 1.300],
    [0.400, 3.500, 4.000, 1.300],
    [1.000, 5.000, 4.000, 1.300],
    [0.430, 6.000, 4.000, 1.400],
    [0.900, 5.000, 4.000, 1.400],
    [0.350, 6.000, 4.000, 1.400],
    [0.400, 6.000, 4.000, 1.400],
    [0.350, 3.500, 4.000, 1.300],
    [0.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.000],
]
# MMFF94力场计算所用的常数项的值
ff_param_const = {
    'power': 0.25,
    'B': 0.2,
    'Beta': 12.0,
    'DARAD': 0.8,
    'DAEPS': 0.5,
    'elec_const': 332.0716,
    'cut_off': 30.0
}

num = 0

use_gpu = torch.cuda.is_available()


def get_device():
    device = "cpu"
    if use_gpu:
        print("GPU is available on this device!")
        device = "cuda"
    else:
        print("CPU is available on this device!")
    return device


device = get_device()


# device = "cpu"
# 从文件夹中读取.sdf数据

def read_mol(path):
    max_heavy_atoms = 0
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]  # 目录下所有文件夹的绝对路径
    mols = []
    labels = []
    j = 0
    k = 0
    for idx, folder in enumerate(cate):
        for suppl_road in glob.glob(folder + '/*.sdf'):  # 获取指定目录下的所有分子文件
            # number = re.search(r'(\d+)\.sdf', suppl_road).group(1)
            suppl = Chem.SDMolSupplier(suppl_road)  # suppl = Chem.SDMolSupplier(suppl_path, removeHs=False)

            for mol in suppl:
                if mol is None:
                    continue
                conf = mol.GetConformer()  # Get the 3D conformer
                # if conf is None:
                #     k+=1
                #     print(f'没有三维结构，跳过了{k}个分子')
                #     continue  # Skip molecules with no 3D conformer
                coords = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
                z_coords = [coord.z for coord in coords]
                if all(z == 0 for z in z_coords):  # 如果所有Z坐标为0，则跳过该分子
                    k += 1
                    print(f'没有三维结构，跳过了{k}个分子')
                    continue
                heavy_atom_count = mol.GetNumHeavyAtoms()
                # 检查是否为最大重原子数
                # if heavy_atom_count > max_heavy_atoms:
                #     max_heavy_atoms = heavy_atom_count
                if heavy_atom_count <= 4:
                    j += 1
                    print('删除了分子{}，因为它的重原子数小于等于4,目前为止共删除了{}个分子'.format(idx,j))
                    continue
                mols.append(mol)
                labels.append(idx)
        num_mols = len(mols)
        print(f'mols 列表中有 {num_mols} 个 mol, 跳过了{k}个分子')
                # print(f"包含最多重原子的分子的重原子数为: {max_heavy_atoms}")

    return np.asarray(mols), np.asarray(labels, np.int32)

# 计算分子中各个原子的部分电荷，取出各个原子的坐标、原子类型
def from_mol_to_array(mol):
    # initialization
    gs_charge, atom_type, pos, nums_atoms = [], [], [], []
    mmff_prop = ChemicalForceFields.MMFFGetMoleculeProperties(mol)
    # print(mmff_prop)
    AllChem.ComputeGasteigerCharges(mol)
    nums_atoms.append(mol.GetNumAtoms())
    # get charge, atom type, 3D coordinates
    for i in range(mol.GetNumAtoms()):
        # get charge
        gs_charge_i = float(mol.GetAtomWithIdx(i).GetProp('_GasteigerCharge'))
        # get atom type
        atom_type_i = mmff_prop.GetMMFFAtomType(i) - 1

        # get coordinate
        pos_i = mol.GetConformer().GetAtomPosition(i)
        pos_x_i, pos_y_i, pos_z_i = pos_i.x, pos_i.y, pos_i.z

        gs_charge.append(gs_charge_i)
        atom_type.append(atom_type_i)
        pos.append([pos_x_i, pos_y_i, pos_z_i])

    return gs_charge, atom_type, pos, nums_atoms

# def from_array_to_ff_batch(b, gs_charge, atom_type, pos, device, mol):
def from_array_to_ff_batch(gs_charge, atom_type, pos, device, mol, result_dict, b):
    # 初始化矩阵
    vdw_matrix_1 = [[0.0] * b for _ in range(b)]
    ele_matrix_1 = [[0.0] * b for _ in range(b)]
    for i, mol1 in enumerate(atom_type):
        for j, mol2 in enumerate(atom_type):

            if i == j or mol.GetBondBetweenAtoms(i, j) != None:
                continue  # 跳过自身原子及有共价键链接的分子
            pos = torch.tensor(pos, device=device, dtype=torch.float32)
            pos1, pos2 = pos[i], pos[j]
            r_ij = torch.sqrt(torch.sum((pos1 - pos2) ** 2))
            if 2 <= r_ij < 4 and mol1 is not None and mol2 is not None:
                # 计算原子间的相互作用能
                nd_ff_param = torch.tensor(ff_param, device=device, dtype=torch.float32)
                gs_charge = torch.tensor(gs_charge, device=device, dtype=torch.float32)
                gs_charge1, gs_charge2 = gs_charge[i], gs_charge[j]
                atom_type1 = nd_ff_param[mol1, :]
                atom_type2 = nd_ff_param[mol2, :]  # 根据索引取得对应的坐标、部分电荷和原子类型对应的参数

                # 计算原子之间的距离

                #  公式的代码实现
                R_i = atom_type1[2] * (atom_type1[0] ** ff_param_const['power'])
                R_j = atom_type2[2] * (atom_type2[0] ** ff_param_const['power'])
                gamma_ij = (R_i - R_j) / (R_i + R_j)
                f = ff_param_const['B'] * (1 - torch.exp(- ff_param_const['Beta'] * gamma_ij * gamma_ij))
                R_ij = ff_param_const['DAEPS'] * (R_i + R_j) * (1.0 + f)
                e_ij = (181.16 *
                        atom_type1[3] * atom_type2[3] *
                        atom_type1[0] * atom_type2[0]) / \
                       (((atom_type1[0] / atom_type1[1]) ** 0.5 +
                         torch.sqrt(atom_type2[0] / atom_type2[1])) * (R_ij ** 6))

                E_vdw = e_ij * ((1.07 * R_ij / (r_ij + 0.07 * R_ij)) ** 7) * (
                        1.12 * (R_ij ** 7) / (r_ij ** 7 + 0.12 * R_ij) - 2.0)
                E_ele = ff_param_const['elec_const'] * gs_charge1 * gs_charge2 / r_ij

                #  将计算的值赋给矩阵中对应位置的值
                x_a = result_dict[mol1]
                x_b = result_dict[mol2]
                if ele_matrix_1[x_a][x_b] == 0:
                    ele_matrix_1[x_a][x_b] = E_ele
                else:
                    ele_matrix_1[x_a][x_b] += E_ele

                if vdw_matrix_1[x_a][x_b] == 0:
                    vdw_matrix_1[x_a][x_b] = E_vdw
                else:
                    vdw_matrix_1[x_a][x_b] += E_vdw

    vdw_matrix_1, ele_matrix_1 = torch.tensor(vdw_matrix_1, device=device, dtype=torch.float32), \
                                 torch.tensor(ele_matrix_1, device=device, dtype=torch.float32)
    # vdw_matrix_1, ele_matrix_1 = torch.abs(vdw_matrix_1), torch.abs(ele_matrix_1)
    ele_matrix_1 = torch.clamp(ele_matrix_1, min=-100, max=100)
    vdw_matrix_1 = torch.clamp(vdw_matrix_1, min=-300, max=300)

    vdw_matrix_2 = [[0.0] * b for _ in range(b)]
    ele_matrix_2 = [[0.0] * b for _ in range(b)]
    for i, mol1 in enumerate(atom_type):
        for j, mol2 in enumerate(atom_type):

            if i == j or mol.GetBondBetweenAtoms(i, j) != None:
                continue  # 跳过自身原子及有共价键链接的分子
            pos = torch.tensor(pos, device=device, dtype=torch.float32)
            pos1, pos2 = pos[i], pos[j]
            r_ij = torch.sqrt(torch.sum((pos1 - pos2) ** 2))
            if 4 <= r_ij < 6 and mol1 is not None and mol2 is not None:
                # 计算原子间的相互作用能
                nd_ff_param = torch.tensor(ff_param, device=device, dtype=torch.float32)
                gs_charge = torch.tensor(gs_charge, device=device, dtype=torch.float32)
                gs_charge1, gs_charge2 = gs_charge[i], gs_charge[j]
                atom_type1 = nd_ff_param[mol1, :]
                atom_type2 = nd_ff_param[mol2, :]  # 根据索引取得对应的坐标、部分电荷和原子类型对应的参数

                # 计算原子之间的距离

                #  公式的代码实现
                R_i = atom_type1[2] * (atom_type1[0] ** ff_param_const['power'])
                R_j = atom_type2[2] * (atom_type2[0] ** ff_param_const['power'])
                gamma_ij = (R_i - R_j) / (R_i + R_j)
                f = ff_param_const['B'] * (1 - torch.exp(- ff_param_const['Beta'] * gamma_ij * gamma_ij))
                R_ij = ff_param_const['DAEPS'] * (R_i + R_j) * (1.0 + f)
                e_ij = (181.16 *
                        atom_type1[3] * atom_type2[3] *
                        atom_type1[0] * atom_type2[0]) / \
                       (((atom_type1[0] / atom_type1[1]) ** 0.5 +
                         torch.sqrt(atom_type2[0] / atom_type2[1])) * (R_ij ** 6))

                E_vdw = e_ij * ((1.07 * R_ij / (r_ij + 0.07 * R_ij)) ** 7) * (
                        1.12 * (R_ij ** 7) / (r_ij ** 7 + 0.12 * R_ij) - 2.0)
                E_ele = ff_param_const['elec_const'] * gs_charge1 * gs_charge2 / r_ij

                #  将计算的值赋给矩阵中对应位置的值
                x_a = result_dict[mol1]
                x_b = result_dict[mol2]
                if ele_matrix_2[x_a][x_b] == 0:
                    ele_matrix_2[x_a][x_b] = E_ele
                else:
                    ele_matrix_2[x_a][x_b] += E_ele

                if vdw_matrix_2[x_a][x_b] == 0:
                    vdw_matrix_2[x_a][x_b] = E_vdw
                else:
                    vdw_matrix_2[x_a][x_b] += E_vdw

    vdw_matrix_2, ele_matrix_2 = torch.tensor(vdw_matrix_2, device=device, dtype=torch.float32), \
                                 torch.tensor(ele_matrix_2, device=device, dtype=torch.float32)
    # vdw_matrix_2, ele_matrix_2 = torch.abs(vdw_matrix_2),torch.abs(ele_matrix_2)
    ele_matrix_2 = torch.clamp(ele_matrix_2, min=-100, max=100)
    vdw_matrix_2 = torch.clamp(vdw_matrix_2, min=-300, max=300)

    vdw_matrix_3 = [[0.0] * b for _ in range(b)]
    ele_matrix_3 = [[0.0] * b for _ in range(b)]
    for i, mol1 in enumerate(atom_type):
        for j, mol2 in enumerate(atom_type):

            if i == j or mol.GetBondBetweenAtoms(i, j) != None:
                continue  # 跳过自身原子及有共价键链接的分子
            pos = torch.tensor(pos, device=device, dtype=torch.float32)
            pos1, pos2 = pos[i], pos[j]
            r_ij = torch.sqrt(torch.sum((pos1 - pos2) ** 2))
            if 6 <= r_ij < 8 and mol1 is not None and mol2 is not None:
                # 计算原子间的相互作用能
                nd_ff_param = torch.tensor(ff_param, device=device, dtype=torch.float32)
                gs_charge = torch.tensor(gs_charge, device=device, dtype=torch.float32)
                gs_charge1, gs_charge2 = gs_charge[i], gs_charge[j]
                atom_type1 = nd_ff_param[mol1, :]
                atom_type2 = nd_ff_param[mol2, :]  # 根据索引取得对应的坐标、部分电荷和原子类型对应的参数

                # 计算原子之间的距离

                #  公式的代码实现
                R_i = atom_type1[2] * (atom_type1[0] ** ff_param_const['power'])
                R_j = atom_type2[2] * (atom_type2[0] ** ff_param_const['power'])
                gamma_ij = (R_i - R_j) / (R_i + R_j)
                f = ff_param_const['B'] * (1 - torch.exp(- ff_param_const['Beta'] * gamma_ij * gamma_ij))
                R_ij = ff_param_const['DAEPS'] * (R_i + R_j) * (1.0 + f)
                e_ij = (181.16 *
                        atom_type1[3] * atom_type2[3] *
                        atom_type1[0] * atom_type2[0]) / \
                       (((atom_type1[0] / atom_type1[1]) ** 0.5 +
                         torch.sqrt(atom_type2[0] / atom_type2[1])) * (R_ij ** 6))

                E_vdw = e_ij * ((1.07 * R_ij / (r_ij + 0.07 * R_ij)) ** 7) * (
                        1.12 * (R_ij ** 7) / (r_ij ** 7 + 0.12 * R_ij) - 2.0)
                E_ele = ff_param_const['elec_const'] * gs_charge1 * gs_charge2 / r_ij

                #  将计算的值赋给矩阵中对应位置的值
                x_a = result_dict[mol1]
                x_b = result_dict[mol2]
                if ele_matrix_3[x_a][x_b] == 0:
                    ele_matrix_3[x_a][x_b] = E_ele
                else:
                    ele_matrix_3[x_a][x_b] += E_ele

                if vdw_matrix_3[x_a][x_b] == 0:
                    vdw_matrix_3[x_a][x_b] = E_vdw
                else:
                    vdw_matrix_3[x_a][x_b] += E_vdw

    vdw_matrix_3, ele_matrix_3 = torch.tensor(vdw_matrix_3, device=device, dtype=torch.float32), \
                                 torch.tensor(ele_matrix_3, device=device, dtype=torch.float32)
    # vdw_matrix_3, ele_matrix_3 = torch.abs(vdw_matrix_3), torch.abs(ele_matrix_3)
    ele_matrix_3 = torch.clamp(ele_matrix_3, min=-100, max=100)
    vdw_matrix_3 = torch.clamp(vdw_matrix_3, min=-300, max=300)

    vdw_matrix_4 = [[0.0] * b for _ in range(b)]
    ele_matrix_4 = [[0.0] * b for _ in range(b)]
    for i, mol1 in enumerate(atom_type):
        for j, mol2 in enumerate(atom_type):

            if i == j or mol.GetBondBetweenAtoms(i, j) != None:
                continue  # 跳过自身原子及有共价键链接的分子
            pos = torch.tensor(pos, device=device, dtype=torch.float32)
            pos1, pos2 = pos[i], pos[j]
            r_ij = torch.sqrt(torch.sum((pos1 - pos2) ** 2))
            if 8 <= r_ij and mol1 is not None and mol2 is not None:
                # 计算原子间的相互作用能
                nd_ff_param = torch.tensor(ff_param, device=device, dtype=torch.float32)
                gs_charge = torch.tensor(gs_charge, device=device, dtype=torch.float32)
                gs_charge1, gs_charge2 = gs_charge[i], gs_charge[j]
                atom_type1 = nd_ff_param[mol1, :]
                atom_type2 = nd_ff_param[mol2, :]  # 根据索引取得对应的坐标、部分电荷和原子类型对应的参数

                # 计算原子之间的距离

                #  公式的代码实现
                R_i = atom_type1[2] * (atom_type1[0] ** ff_param_const['power'])
                R_j = atom_type2[2] * (atom_type2[0] ** ff_param_const['power'])
                gamma_ij = (R_i - R_j) / (R_i + R_j)
                f = ff_param_const['B'] * (1 - torch.exp(- ff_param_const['Beta'] * gamma_ij * gamma_ij))
                R_ij = ff_param_const['DAEPS'] * (R_i + R_j) * (1.0 + f)
                e_ij = (181.16 *
                        atom_type1[3] * atom_type2[3] *
                        atom_type1[0] * atom_type2[0]) / \
                       (((atom_type1[0] / atom_type1[1]) ** 0.5 +
                         torch.sqrt(atom_type2[0] / atom_type2[1])) * (R_ij ** 6))

                E_vdw = e_ij * ((1.07 * R_ij / (r_ij + 0.07 * R_ij)) ** 7) * (
                        1.12 * (R_ij ** 7) / (r_ij ** 7 + 0.12 * R_ij) - 2.0)
                E_ele = ff_param_const['elec_const'] * gs_charge1 * gs_charge2 / r_ij

                #  将计算的值赋给矩阵中对应位置的值
                x_a = result_dict[mol1]
                x_b = result_dict[mol2]
                if ele_matrix_4[x_a][x_b] == 0:
                    ele_matrix_4[x_a][x_b] = E_ele
                else:
                    ele_matrix_4[x_a][x_b] += E_ele

                if vdw_matrix_4[x_a][x_b] == 0:
                    vdw_matrix_4[x_a][x_b] = E_vdw
                else:
                    vdw_matrix_4[x_a][x_b] += E_vdw

    vdw_matrix_4, ele_matrix_4 = torch.tensor(vdw_matrix_4, device=device, dtype=torch.float32), \
                                 torch.tensor(ele_matrix_4, device=device, dtype=torch.float32)
    # vdw_matrix_4, ele_matrix_4 = torch.abs(vdw_matrix_4), torch.abs(ele_matrix_4)
    ele_matrix_4 = torch.clamp(ele_matrix_4, min=-100, max=100)
    vdw_matrix_4 = torch.clamp(vdw_matrix_4, min=-300, max=300)

    E1 = torch.stack((vdw_matrix_1, vdw_matrix_2, vdw_matrix_3, vdw_matrix_4), 2).to(device)
    # print(E1.shape)
    E2 = torch.stack((ele_matrix_1, ele_matrix_2, ele_matrix_3, ele_matrix_4), 2).to(device)
    E = torch.cat((E1, E2), 2).to(device).permute(2, 0, 1)
    # E = torch.empty((8, b, b)).to(device)
    # 交替排列
    # E[0::2] = E1  # 填充偶数位置
    # E[1::2] = E2
    return E.squeeze()

def cal_angle(a, b, c):
    ba = a - b
    bc = c - b

    dot = torch.matmul(ba.unsqueeze(-1).transpose(-2, -1), bc.unsqueeze(-1))
    cosine_angle = dot.squeeze(-1) / (
                torch.norm(ba, p=2, dim=1).reshape(-1, 1) * torch.norm(bc, p=2, dim=1).reshape(-1, 1))
    cosine_angle = torch.where(torch.logical_or(cosine_angle > 1, cosine_angle < -1), torch.round(cosine_angle),
                               cosine_angle)
    angle = torch.arccos(cosine_angle)

    return angle

def cal_dihedral(a, b, c, d):
    ab = a - b
    cb = c - b
    dc = d - c

    cb /= torch.norm(cb, p=2, dim=1).reshape(-1, 1)
    v = ab - torch.matmul(ab.unsqueeze(-1).transpose(-2, -1), cb.unsqueeze(-1)).squeeze(-1) * cb
    w = dc - torch.matmul(dc.unsqueeze(-1).transpose(-2, -1), cb.unsqueeze(-1)).squeeze(-1) * cb
    x = torch.matmul(v.unsqueeze(-1).transpose(-2, -1), w.unsqueeze(-1)).squeeze(-1)
    y = torch.matmul(torch.cross(cb, v).unsqueeze(-1).transpose(-2, -1), w.unsqueeze(-1)).squeeze(-1)

    return torch.atan2(y, x)

def calculate_covalent_interactions(molecule):
    bonds = molecule.GetBonds()
    interactions = []
    for bond in bonds:
        atom1 = bond.GetBeginAtomIdx()
        atom2 = bond.GetEndAtomIdx()
        distance = molecule.GetConformer().GetAtomPosition(atom1).Distance(
            molecule.GetConformer().GetAtomPosition(atom2))
        interactions.append((atom1, atom2, 'covalent', distance))
    return interactions

def calculate_angle_bending_interactions(molecule):
    angles = set()
    positions = molecule.GetConformer().GetPositions()
    for bond in molecule.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()

        # 遍历 atom1 的邻居（排除 atom2）
        for neighbor in atom1.GetNeighbors():
            if neighbor.GetIdx() != atom2.GetIdx():
                a = torch.tensor(positions[neighbor.GetIdx()])
                b = torch.tensor(positions[atom1.GetIdx()])
                c = torch.tensor(positions[atom2.GetIdx()])
                angle = cal_angle(a.unsqueeze(0), b.unsqueeze(0), c.unsqueeze(0)).item()
                angles.add((neighbor.GetIdx(), atom1.GetIdx(), atom2.GetIdx(), 'angle', angle))

        # 遍历 atom2 的邻居（排除 atom1）
        for neighbor in atom2.GetNeighbors():
            if neighbor.GetIdx() != atom1.GetIdx():
                a = torch.tensor(positions[atom1.GetIdx()])
                b = torch.tensor(positions[atom2.GetIdx()])
                c = torch.tensor(positions[neighbor.GetIdx()])
                angle = cal_angle(a.unsqueeze(0), b.unsqueeze(0), c.unsqueeze(0)).item()
                angles.add((atom1.GetIdx(), atom2.GetIdx(), neighbor.GetIdx(), 'angle', angle))

    return list(angles)

def calculate_torsion_interactions(molecule):
    torsions = []
    positions = molecule.GetConformer().GetPositions()
    for bond in molecule.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()

        # 遍历 atom1 的邻居，排除 atom2
        for neighbor1 in atom1.GetNeighbors():
            if neighbor1.GetIdx() != atom2.GetIdx():

                # 遍历 atom2 的邻居，排除 atom1 和 neighbor1
                for neighbor2 in atom2.GetNeighbors():
                    if neighbor2.GetIdx() != atom1.GetIdx() and neighbor2.GetIdx() != neighbor1.GetIdx():
                        # 计算二面角
                        a = torch.tensor(positions[neighbor1.GetIdx()])
                        b = torch.tensor(positions[atom1.GetIdx()])
                        c = torch.tensor(positions[atom2.GetIdx()])
                        d = torch.tensor(positions[neighbor2.GetIdx()])
                        torsion = cal_dihedral(a.unsqueeze(0), b.unsqueeze(0), c.unsqueeze(0), d.unsqueeze(0)).item()
                        torsions.append((neighbor1.GetIdx(), atom1.GetIdx(), atom2.GetIdx(), neighbor2.GetIdx(),
                                         'torsion', torsion))

    return torsions

def create_index_mapping(elements):
    """创建唯一标识符到索引的映射"""
    mapping = {}
    idx = 0
    for element in elements:
        if element not in mapping:
            mapping[element] = idx
            idx += 1
    return mapping
def generate_key_for_edge(atom1, atom2):
    """生成一个键的标识符（类型），确保原子序号顺序一致"""
    return tuple(sorted([atom1, atom2]))

def generate_key_for_angle(atom1, atom2, atom3):
    """生成一个角度的标识符（类型），确保原子序号顺序一致"""
    return tuple(sorted([atom1, atom2, atom3]))

def ff_inter_juzhen(edge_index, angle_index, covalent_interactions, angle_bending_interactions, torsion_interactions, b, device='cpu'):
    ff_matrix_1 = [[0.0] * b for _ in range(b)]
    ff_matrix_2 = [[0.0] * b for _ in range(b)]
    ff_matrix_3 = [[0.0] * b for _ in range(b)]

    for cov in covalent_interactions:
        g, h, _, cov_value = cov
        if ff_matrix_1[g][h] == 0:
            ff_matrix_1[g][h] = cov_value
        else:
            ff_matrix_1[g][h] = (ff_matrix_1[g][h] + cov_value) / 2

    for angle in angle_bending_interactions:
        i, j, k, _, angle_value = angle
        edge_key1 = generate_key_for_edge(i, j)
        edge_key2 = generate_key_for_edge(j, k)

        row, col = edge_index[edge_key1], edge_index[edge_key2]

        if ff_matrix_2[row][col] == 0:
            ff_matrix_2[row][col] = angle_value
        else:
            print('角度重合')
            ff_matrix_2[row][col] = (ff_matrix_2[row][col] + angle_value) / 2

    for tors in torsion_interactions:
        l, m, n, o, _, tors_value = tors
        angle_key1 = generate_key_for_angle(l, m, n)
        angle_key2 = generate_key_for_angle(m, n, o)

        row, col = angle_index[angle_key1], angle_index[angle_key2]

        if ff_matrix_3[row][col] == 0:
            ff_matrix_3[row][col] = tors_value
        else:
            print("二面角重合")
            ff_matrix_3[row][col] = (ff_matrix_3[row][col] + tors_value) / 2

    # 将嵌套列表转换为张量
    ff_matrix_1 = torch.tensor(ff_matrix_1, device=device)
    ff_matrix_2 = torch.tensor(ff_matrix_2, device=device)
    ff_matrix_3 = torch.tensor(ff_matrix_3, device=device)
    E3 = torch.stack((ff_matrix_1, ff_matrix_2, ff_matrix_3), 2).permute(2, 0, 1)
    # E3 = torch.stack((ff_matrix_1, ff_matrix_2), 2).permute(2, 0, 1)
    return E3

def edge_angle(covalent_interactions, torsion_interactions):
    unique_edges = {generate_key_for_edge(cov[0], cov[1]) for cov in covalent_interactions}
    unique_angles = set()
    for tors in torsion_interactions:
        # 每个 torsion_interactions 包含两个角
        angle1 = generate_key_for_angle(tors[0], tors[1], tors[2])
        angle2 = generate_key_for_angle(tors[1], tors[2], tors[3])
        unique_angles.add(angle1)
        unique_angles.add(angle2)
    edge_index = create_index_mapping(unique_edges)
    angle_index = create_index_mapping(unique_angles)
    return edge_index, angle_index
def save_e_values_to_json(train_path, output_path, txt_name, num):
    X_train, Y_train = read_mol(train_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    b, result_dict = new_atom_type(X_train, Y_train)
    # print('b = ', b)
    b = 100
    ed = 0
    ee = 0
    ef = 0
    for idx, (mol, y_value) in enumerate(zip(X_train, Y_train)):

        mmff_prop = ChemicalForceFields.MMFFGetMoleculeProperties(mol)
        if mmff_prop is None:
            print('跳过了分子：', idx)
            ed += 1
            print("因为分子错误而跳过的分子数量有：", ed)
            continue
        heavy_atom_count = mol.GetNumHeavyAtoms()
        # 判断重原子数是否大于40
        if heavy_atom_count > b:
            print(idx,'的重原子数为',heavy_atom_count, '重原子数大于', b)
            ee += 1
            print("因为分子太大而跳过的分子数量有：", ee)
            continue
        gs_charge, atom_type, pos, nums_atoms = from_mol_to_array(mol)
        covalent_interactions = calculate_covalent_interactions(mol)
        angle_bending_interactions = calculate_angle_bending_interactions(mol)
        torsion_interactions = calculate_torsion_interactions(mol)
        edge_index, angle_index = edge_angle(covalent_interactions, torsion_interactions)
        print('有{}条边，有{}个角'.format(len(edge_index), len(angle_index)))
        if len(angle_index) > b:
            print(idx, '重原子数是',heavy_atom_count, '角数大于', b)
            ef += 1
            print("因为角数太多而跳过的分子数量有：", ef)
            continue
        E1 = ff_inter_juzhen(edge_index, angle_index, covalent_interactions, angle_bending_interactions, torsion_interactions, b)
        E2 = from_array_to_ff_batch(gs_charge, atom_type, pos, device, mol, result_dict, b)
        E1 = E1.to(device)
        E2 = E2.to(device)
        E = torch.cat((E2, E1), dim=0)
        print(E.shape)
        # 将E值从Tensor转换为NumPy数组
        E_np = E.detach().cpu().numpy()
        Y_int = int(y_value)

        # 添加E值和Y值到字典中
        data_dict = {
            'E': E_np.tolist(),
            'Y': Y_int
        }

        # 将字典保存为JSON文件
        with open(os.path.join(output_path, f'{txt_name}_{b}_{num}.json'), 'w') as json_file:
            json.dump(data_dict, json_file)
        num = num + 1


cyp_list = ['ames_275']
for file_name in cyp_list:
    train_path = '/home/cpj/ybj/force_file_data/ames_dataset275/'
    output_path = '/home/cpj/ybj/force_file_data/ames_all_json/{}_11td_100_json'.format(file_name)
    txt_name = '{}'.format(file_name)
    save_e_values_to_json(train_path, output_path, txt_name, num)