import numpy as np

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

def viterbi(seq, T, E, pi):
    states = {0: 'E', 1: '5', 2: 'I', 3: 'End'}
    n_states = T.shape[0]
    n_obs = len(seq)
    
    # V — максимальные вероятности, ptr — карта обратного пути
    V = np.zeros((n_states, n_obs))
    ptr = np.zeros((n_states, n_obs), dtype=int) 
    
    # Шаг 1: Инициализация (t=0)
    for s in range(n_states):
        V[s, 0] = pi[s] + E[s, seq[0]]
        
    for t in range(1, n_obs):
        for s in range(n_states):
            # дотягиваемся до s-того состояния из всевозможных на t-1 шаге
            probabilities = V[:, t-1] + T[:, s]
            
            # Находим индекс лучшего "предка"
            best_prev_state = np.argmax(probabilities)
            
            # Сохраняем "предка" в ptr и саму вероятность в V
            ptr[s, t] = best_prev_state
            V[s, t] = probabilities[best_prev_state] + E[s, seq[t]]

    # Шаг 3: Восстановление пути (Backtracking)
    best_path_idx = np.zeros(n_obs, dtype=int)
    
    # 3.1 Находим самое вероятное состояние в самом конце
    best_path_idx[-1] = np.argmax(V[:, -1])
    
    # 3.2 Идем назад от предпоследнего шага до начала
    for t in range(n_obs - 2, -1, -1):
        # Чтобы узнать, где мы были в 't', смотрим, 
        # кто был предком текущего состояния в момент 't+1'
        current_state_at_next_step = best_path_idx[t+1]
        best_path_idx[t] = ptr[current_state_at_next_step, t+1]

    result_str = ''.join([states[s] for s in best_path_idx])
    return result_str, V

path_str, delta_matrix = viterbi(seq, T, E, pi)

print("Анализируемая строка: ", seq_str)
print("Найденный путь:       ", path_str)