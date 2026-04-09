import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
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

def backward(seq, E, T):
    """
    Вычисляет логарифмические вероятности Backward.
    """
    n_states = T.shape[0]
    n_obs = len(seq)
    
    # Создаем матрицу B и заполняем её очень маленькими числами (log 0)
    B = np.full((n_states, n_obs), NEG_INF)

    # Шаг 1: Инициализация (последний столбец)
    # В обычных вероятностях это 1, в логарифмах log(1) = 0
    for s in range(n_states):
        B[s, -1] = 0 
    
    # Шаг 2: Обратный проход (от n_obs-2 до 0)
    for t in range(n_obs - 2, -1, -1):
        for s in range(n_states):
            # Нам нужно собрать сумму по всем возможным СЛЕДУЮЩИМ состояниям (next_s)
            # Формула для каждого пути: log_trans + log_emission + log_backward_next
            work_probs = T[s, :] + E[:, seq[t+1]] + B[:, t+1]
            
            # Суммируем эти пути через logsumexp
            B[s, t] = logsumexp(work_probs)
            
    return B

def forward(seq, E, T, pi):
    n_states = T.shape[0]
    n_obs = len(seq)
    F = np.zeros((n_states, n_obs))

    for s in range(n_states):
        F[s, 0] = pi[s] + E[s, seq[0]]
    
    for t in range(1, n_obs):
        for s in range(n_states):
            F[s, t] = logsumexp(F[:, t-1] + T[:, s]) + E[s, seq[t]] # проблема суммирования вероятностей - P1 + P2 != logP1 + logP2
    log_entire_probability = logsumexp(F[:, -1])
    return log_entire_probability, F

def aposterior_decoding(F, B, log_entire_probability):
    matrix = np.zeros_like(F)

    n_states = matrix.shape[0]
    n_obs = matrix.shape[1]

    for t in range(n_obs):
        for s in range(n_states):
            matrix[s, t] = np.exp((F[s, t] + B[s,t])  - log_entire_probability)
    return matrix


log_entire_probability, F = forward(seq, E, T, pi)
B = backward(seq, E, T)


one_matrix_to_rule_them_all = aposterior_decoding(F, B, log_entire_probability)

E_chances = one_matrix_to_rule_them_all[0, :] # Экзон
s_chances = one_matrix_to_rule_them_all[1, :] # Сайт сплайсинга (5')
I_chances = one_matrix_to_rule_them_all[2, :] # Интрон


ts = np.arange(len(seq))

plt.figure(figsize=(16, 9))

# Строим графики для каждого состояния
plt.plot(ts, E_chances, label='Экзон (E)', color='green', linewidth=2)
plt.plot(ts, s_chances, label='Сайт сплайсинга (5)', color='orange', linewidth=2)
plt.plot(ts, I_chances, label='Интрон (I)', color='red', linewidth=2)

# Настройка осей и легенды
plt.title("Апостериорное декодирование: вероятность состояний для каждого нуклеотида", fontsize=16)
plt.xlabel("Позиция в последовательности", fontsize=12)
plt.ylabel("Вероятность P(state_t | seq)", fontsize=12)

# Заменяем цифры на оси X на сами нуклеотиды из твоей строки
plt.xticks(ts, list(seq_str))

# Добавляем сетку и легенду
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right', fontsize=12)

# Устанавливаем границы по вертикали от 0 до 1 (так как это вероятности)
plt.ylim(-0.05, 1.05)

# Показываем результат
plt.show()

