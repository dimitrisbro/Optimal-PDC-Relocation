import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pulp import *
from collections import Counter


def pdcPlacement(branches, bus, pmus, distMatrix, threshold):
    fig, ax = plt.subplots(2)
    G = nx.from_edgelist(branches)
    my_pos = nx.spring_layout(G, seed=150)
    nx.draw(G, with_labels=True, pos=my_pos,ax=ax[0])

    # dist = np.zeros((bus, bus))
    # for h in range(bus):
    #     d = nx.single_source_dijkstra_path_length(G, h + 1)
    #     for j in range(bus):
    #         dist[h][j] = d[j + 1]



    sets = []
    for i in range(len(pmus)):
        s = []
        for h in range(bus):
            if distMatrix[i][h] <= threshold:
                # s.append(h + 1)
                s.append(h)
        sets.append(s)

    # for i in sets:
    #     print(i)

    A = np.zeros((len(pmus), bus))
    for i in range(len(sets)):
        A[i][sets[i]] = 1

    # print(A)
    c = np.ones(bus)
    b = np.ones(bus)

    prob = LpProblem("PMU-Placement", LpMinimize)

    m, n = len(A), len(A[0])
    COLS = range(n)
    ROWS = range(m)

    x = LpVariable.dicts('x', COLS, lowBound=0, upBound=1, cat='Integer')

    prob += lpSum(c[j] * x[j] for j in COLS), "obj"

    for i in ROWS:
        prob += lpSum(A[i][j] * x[j] for j in COLS) >= b[i]

    # print(prob)

    prob.solve()

    print("Status    :", LpStatus[prob.status])
    print("objective =", value(prob.objective))
    pdc = []
    pdcn = []
    for v in prob.variables():
        print(v.name, "=", '{0:.2f}'.format(v.varValue))
        pdc.append(int(v.varValue))
        pdcn.append(int(v.name.split('_')[1])+1)

    pdcValue = [x for _, x in sorted(zip(pdcn, pdc))]

    color = []
    for i in range(bus):
        if pdcValue[i] == 0:
            color.append("blue")
        else:
            color.append("red")

    plt.suptitle("PDC Placement ")
    nx.draw(G, with_labels=True, node_color=color, pos=my_pos, ax=ax[1])
    plt.show()
    return LpStatus[prob.status], value(prob.objective), pdcValue

def pmuPlacement(branches, bus, injection, flow):
    buses = []
    c = np.ones(bus)
    b = []

    for i in range(bus):
        buses.append(i + 1)
    Tpmu = np.zeros((bus, bus))
    np.fill_diagonal(Tpmu, 1)

    for i in range(branches.shape[0]):
        Tpmu[branches[i][0] - 1][branches[i][1] - 1] = 1
        Tpmu[branches[i][1] - 1][branches[i][0] - 1] = 1

    # -----Injection related buses-----
    a = []
    for i in injection:
        # a[l]=(i)
        br = [i]
        for j in range(branches.shape[0]):
            if i == branches[j][0]:
                br.append(branches[j][1])
            elif i == branches[j][1]:
                br.append(branches[j][0])
        br.sort()
        cr = [i, br, len(br) - 1]
        a.append(cr)

    # print("Injection Table\n", a)
    # -----Tmeasurement table-----
    rowsTmeas = len(injection) + len(flow)
    dim = []
    for i in a:
        for j in i[1]:
            dim.append(j)
    for i in flow:
        dim.append(i[0])
        dim.append(i[1])
    # print(dim)
    # print(np.unique(dim))
    var = np.unique(dim)  # variables associated with convetional measurements
    # print("Associated Variables:\n", var)
    # print("rows",len(injection)+len(flow),'Columns',len(np.unique(dim)))
    colsTmeas = len(np.unique(dim))
    Tmeas = np.zeros((rowsTmeas, colsTmeas))

    k = 0
    for i in flow:
        for j in range(colsTmeas):
            if i[0] == var[j] or i[1] == var[j]:
                Tmeas[k][j] = 1
        k = k + 1
        b.append(1)

    # -----I table-----
    # variables not associated with convetional measurements
    nvar = [x for x in set(buses) if x not in var]
    # print("Not Associated Variables:\n", nvar)
    I = np.zeros((len(nvar), len(nvar)))
    np.fill_diagonal(I, 1)
    # print("I:\n", I)

    # -----Measurements association-----
    for i in a:
        flag = 0
        for j in range(len(flow)):
            if i[0] == flow[j][0] or i[0] == flow[j][1]:
                flag = 1
                # print("Association between injection:\n", i, "and flow", flow[j])
                l = i[1] + list(flow[j])
                cr = Counter(zip(l))
                # print(*(k[0] for k, v in c.items() if v == 1))
                lm = [*(r[0] for r, v in cr.items() if v == 1)]
                # print(lm)
                for j in range(len(var)):
                    if var[j] in lm:
                        # print(var[j])
                        Tmeas[k][j] = 1
                k = k + 1
                b.append(i[2] - 2)
        if flag == 0:
            # print("Unassociated Injection:", i[0])
            for m in range(len(var)):
                if var[m] in i[1]:
                    Tmeas[k][m] = 1
            k = k + 1
            b.append(i[2])

    # print("Tmeasurement:\n", Tmeas)

    # -----Tcon table----
    if Tmeas.size == 0:
        Tcon = 1
        b = np.ones(I.shape[0])
    else:
        if I.size == 0:
            Tcon = Tmeas
        else:
            # print(Tmeas.shape)
            # print(I.shape)
            arr1 = np.zeros((I.shape[0], Tmeas.shape[1]))
            arr2 = np.zeros((Tmeas.shape[0], I.shape[1]))
            # print(arr1)
            # print(arr2)
            o = np.concatenate((I, arr1), axis=1)
            f = np.concatenate((arr2, Tmeas), axis=1)
            Tcon = np.concatenate((o, f), axis=0)
            b = np.concatenate((np.ones(I.shape[0]), b))

    # print("Tcon:\n", Tcon)

    # -----Permutation table-----
    kp = 0
    P = np.zeros((bus, bus))
    for j in nvar:
        P[kp][j - 1] = 1
        kp = kp + 1
    for j in var:
        P[kp][j - 1] = 1
        kp = kp + 1

    # print("Permutation Table\n", P)
    A = np.dot(np.dot(Tcon, P), Tpmu)
    # print("A:\n", np.array(A))
    # print("b:\n", b)
    # print("c:\n", c)

    # #'''
    # Optimization for full observability without conventional measurements
    prob = LpProblem("PMU-Placement", LpMinimize)

    m, n = len(A), len(A[0])
    COLS = range(n)
    ROWS = range(m)

    x = LpVariable.dicts('x', COLS, lowBound=0, upBound=1, cat='Integer')

    prob += lpSum(c[j] * x[j] for j in COLS), "obj"

    for i in ROWS:
        prob += lpSum(A[i][j] * x[j] for j in COLS) >= b[i]

    # print(prob)

    prob.solve()

    print("Status    :", LpStatus[prob.status])
    print("objective =", value(prob.objective))
    pmu = []
    pmun = []
    for v in prob.variables():
        print(v.name, "=", '{0:.2f}'.format(v.varValue))
        pmun.append(int(v.name.split('_')[1]) + 1)
        pmu.append(int(v.varValue))

    pmuValue = [x for _, x in sorted(zip(pmun, pmu))]
    return LpStatus[prob.status], value(prob.objective),pmuValue