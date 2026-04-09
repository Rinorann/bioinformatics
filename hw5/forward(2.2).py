import numpy as np
from scipy.special import logsumexp

NEG_INF = -1e12

def safe_log(matrix):#безопасный логарифм
    with np.errstate(divide='ignore'):
        res = np.log(matrix)
    res[np.isinf(res)] = NEG_INF
    return res

T_probs = np.array([
    [0.9, 0.1, 0, 0], 
    [0, 0, 1, 0],
    [0, 0, 0.9, 0.1],
    [0, 0, 0, 0]
])
T = safe_log(T_probs)

E_probs = np.array([
    [0.25, 0.25, 0.25, 0.25],
    [0.05, 0,   0.95, 0],
    [0.4,  0.1, 0.1,  0.4],
    [0,    0,   0,    0] 
]) 
E = safe_log(E_probs)

pi_probs = np.array([1, 0, 0, 0])
pi = safe_log(pi_probs)

nucleotides = {'A': 0, 'C': 1, 'G': 2, 'T': 3} 
seq_str = "CTTCATGTGAAAGCAGACGTAAGTCA"
seq = [nucleotides[s] for s in seq_str]

def forward(seq, E, T, pi):
    n_states = T.shape[0]
    n_obs = len(seq)
    F = np.zeros((n_states, n_obs))

    for s in range(n_states):
        F[s, 0] = pi[s] + E[s, seq[0]]
    
    for t in range(1, n_obs):
        for s in range(n_states):
            F[s, t] = logsumexp(F[:, t-1] + T[:, s]) + E[s, seq[t]] # проблема суммирования вероятностей - P1 + P2 != logP1 + logP2
    last_column = logsumexp(F[:, -1])
    return last_column
print(f' Логарифм вероятности {forward(seq, E, T, pi)}')
print(f' Вероятность {np.exp(forward(seq, E, T, pi))}')