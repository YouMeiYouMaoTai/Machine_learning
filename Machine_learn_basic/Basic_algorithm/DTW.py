from numpy import array, zeros, argmin, inf, equal, ndim

s1 = [1, 6, 3, 4, 3, 5, 5, 4]
s2 = [3, 4, 5, 5, 5, 4]

r, c = len(s1), len(s2)
D0 = zeros((r + 1, c + 1))
D0[0, 1:] = inf
D0[1:, 0] = inf
D1 = D0[1:, 1:]  # 浅复制

for i in range(r):  # 生成原始距离矩阵
    for j in range(c):
        D1[i, j] = abs(s1[i] - s2[j])  # 距离

M = D1.copy()
for i in range(r):  # 代码核心，动态计算最短距离
    for j in range(c):
        D1[i, j] += min(D0[i, j], D0[i, j + 1], D0[i + 1, j])

i, j = array(D0.shape) - 2
# 最短路径
p, q = [i], [j]
while (i > 0 or j > 0):
    # 取最小值
    tb = argmin((D0[i, j], D0[i, j + 1], D0[i + 1, j]))
    if tb == 0:
        i -= 1
        j -= 1
    elif tb == 1:
        i -= 1
    else:
        j -= 1
    p.insert(0, i)
    q.insert(0, j)

# 原始距离矩阵
print("原始矩阵：\n", M)

# 匹配路径过程
print("匹配路径过程；\n", list(zip(p, q)))

# Cost Matrix或者叫累积距离矩阵
print("累计距离矩阵：\n", D1)

print(D1[-1, -1])
# 序列距离
