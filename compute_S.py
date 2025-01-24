import escnn.group as g
import numpy as np

def S(p, q):
    r = p * q
    I_r = np.eye(p*q, dtype=float)
    out = np.concatenate([I_r[i:r:q, :] for i in range(q)], axis=0)
    return out


def TracyS(deg1, deg2):
    Size1 = np.array([0] + list(2 * deg1 + 1))
    Total_Size1 = np.cumsum(Size1)

    Size2 = np.array([0] + list(2 * deg2 + 1))
    Total_Size2 = np.cumsum(Size2)

    # Determine which block q is in
    def locate(q, Total_Size):
        block_index = np.sum(q >= Total_Size)
        res = q - Total_Size[block_index - 1]
        return block_index, res

    def get_index(m):
        p, q = divmod(m, np.sum(Size2))
        block_index1, res1 = locate(p, Total_Size1)
        block_index2, res2 = locate(q, Total_Size2)

        position = Total_Size1[block_index1-1] * np.sum(Size2) +  \
                    Size1[block_index1] * Total_Size2[block_index2-1] + \
                    res1 * Size2[block_index2] + res2
        return position

    def get_index_1(m):
        p, q = divmod(m, np.sum(Size2))
        block_index1, res1 = locate(p, Total_Size1)
        block_index2, res2 = locate(q, Total_Size2)
        position = Total_Size2[block_index2-1] * np.sum(Size1) +  \
                    Size2[block_index2] * Total_Size1[block_index1-1] + \
                    res1 * Size2[block_index2] + res2

        return position

    Permutation = np.zeros((np.sum(Size1) * np.sum(Size2), np.sum(Size1) * np.sum(Size2)), dtype=int)
    for i in range(Permutation.shape[0]):
        Permutation[i, get_index(i)] = 1

    Permutation1 = np.zeros((np.sum(Size1) * np.sum(Size2), np.sum(Size1) * np.sum(Size2)), dtype=int)
    for i in range(Permutation1.shape[0]):
        Permutation1[i, get_index_1(i)] = 1
    return Permutation, Permutation1


def Q(i1, i2, o1, o2):
    """
    Compute the change of basis matrix for SO3xSO3
    :param i1: variable 1 input degree
    :param i2: variable 2 input degree
    :param o1: variable 1 output degree
    :param o2: variable 2 output degree
    :return: the change of basis matrix Q.
    Q.shape=((2*i1 + 1) * (2*o1 + 1) *(2*o2 + 1) *(2*i2 + 1), (2*i1 + 1) * (2*o1 + 1) *(2*o2 + 1) *(2*i2 + 1) )
    """
    r1 = g.so3_group(3)
    variable1_tensor = r1.irreps()[i1].tensor(r1.irreps()[o1]).change_of_basis
    variable2_tensor = r1.irreps()[i2].tensor(r1.irreps()[o2]).change_of_basis

    S_large = np.kron(np.kron(np.eye(i1*2+1, dtype=float), S(o1*2+1, i2*2+1)), np.eye(o2*2+1, dtype=float))
    decomp = S_large @ (np.kron(variable1_tensor, variable2_tensor))
    var1_deg = np.array([i for i in range(np.abs(i1 - o1), i1 + o1 + 1)])
    var2_deg = np.array([i for i in range(np.abs(i2 - o2), i2 + o2 + 1)])
    P_2_first, P_1_first = TracyS(var1_deg, var2_deg)
    S_circle = decomp @ P_2_first # in the direct sum, the second variable changes the fastest
    S_circle_1 = decomp @ P_1_first # in the direct sum, the first variable changes the fastest
    return S_circle, S_circle_1

if __name__ == '__main__':
    import scipy
    r1 = g.so3_group(3)
    r2 = g.so3_group(3)
    r12 = g.direct_product(r1, r2)
    element12 = r12.sample()
    element1, element2 = r12.split_element(element12)

    i1, i2, o1, o2 = 1, 2, 0, 2
    S_circle, S_circle_1 = Q(i1, i2, o1, o2)


    var1_deg = [i for i in range(np.abs(i1-o1), i1+o1+1)]
    var2_deg = [i for i in range(np.abs(i2-o2), i2+o2+1)]

    var1_rep = [r1.irreps()[i](element1) for i in var1_deg]
    var2_rep = [r1.irreps()[i](element2) for i in var2_deg]

    kron_tensor = np.kron(np.kron(r1.irreps()[i1](element1), r1.irreps()[i2](element2)),
                            np.kron(r1.irreps()[o1](element1), r1.irreps()[o2](element2)))

    # fast2
    Kron = []
    for i in var1_rep:
        for j in var2_rep:
            Kron.append(np.kron(i, j))
    fast2 = scipy.linalg.block_diag(*Kron)

    # fast 1
    Kron = []
    for j in var2_rep:
        for i in var1_rep:
            Kron.append(np.kron(i, j))
    fast1 = scipy.linalg.block_diag(*Kron)

    computed_final_2_fast = S_circle.T @ kron_tensor @ S_circle
    computed_final_1_fast = S_circle_1.T @ kron_tensor @ S_circle_1

    print('fast1_Error: {}'.format(np.max(np.abs(computed_final_1_fast - fast1))))
    print('fast2_Error: {}'.format(np.max(np.abs(computed_final_2_fast - fast2))))
