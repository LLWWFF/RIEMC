from itertools import permutations
import itertools
import random
import numpy as np


#获取所有点集
def V_S(n):
    # 生成1到n的数字列表
    numbers = list(range(1, n + 1))
    # 使用permutations函数获取从1到n中取k个数字的排列
    perms = permutations(numbers, n)
    # 将排列转换为列表并返回
    result = list(perms)
    # 将元组转换为整数
    result = [int(''.join(map(str, perm))) for perm in result]

    str_list = [str(element) for element in result]

    return str_list

#两点之间海明距离
def H(str1,str2):
    res = 0
    for i in range(len(str1)):
        if str1[i] != str2[i]:
            res = res + 1
    return res

def quchong(list):
    unique_list = []
    seen_sublists = set()

    for sublist in list:
        # 将子列表中的元素排序并连接为字符串
        sorted_str = ''.join(sorted(sublist))

        # 检查字符串是否已经出现过
        if sorted_str not in seen_sublists:
            unique_list.append(sorted(sublist))
            seen_sublists.add(sorted_str)
    return unique_list

#根据连边规则连边
def E(my_list,n):
    my_E_list = []
    for vertex1 in my_list:
        for vertex2 in my_list:
            if vertex1 != vertex2:
                #判断swapped_vertex1的0位与swapped_vertex2的1-size(swapped_vertex2)位是否相同
                for i in range(n):
                    if vertex1[0] != vertex2[0]:
                        if vertex1[0] == vertex2[i]:
                            #若相同则再次判断swapped_vertex2的0位是否与swapped_vertex1的第i位相同
                            if vertex2[0] == vertex1[i]:
                                if H(vertex1,vertex2) == 2:
                                    my_E_list.append([vertex1, vertex2])

    return quchong(my_E_list)


def adjacency_matrix(vertices, edges):
    # 获取点的数量
    num_vertices = len(vertices)

    # 创建一个初始全零的邻接矩阵
    adj_matrix = [[0] * num_vertices for _ in range(num_vertices)]

    # 填充邻接矩阵
    for edge in edges:
        vertex1, vertex2 = edge
        # 由于是无向图，所以边是双向的，需要在两个方向上标记
        index1 = vertices.index(vertex1)
        index2 = vertices.index(vertex2)
        adj_matrix[index1][index2] = 1
        adj_matrix[index2][index1] = 1
    return adj_matrix

def generate_adjacency_matrix(n):
    vertices = V_S(n)
    edges = E(vertices, n)
    matrix = adjacency_matrix(vertices, edges)
    return matrix
def print_adjacency_matrix(matrix):
    for row in matrix:
        print(row)

def get_edges_neighbors_count(adj_matrix, num_vertices):
    # Convert adjacency matrix to set of edges
    edge_set = set()
    for i in range(len(adj_matrix)):
        for j in range(i + 1, len(adj_matrix)):
            if adj_matrix[i][j] == 1:
                edge_set.add((i, j))

    # Randomly select num_vertices edges without duplicate nodes
    selected_edges = set()
    selected_nodes = set()

    while len(selected_edges) < num_vertices:
        edge = random.choice(list(edge_set))
        node1, node2 = edge

        if node1 not in selected_nodes and node2 not in selected_nodes \
                and all((node1, n) not in selected_edges and (n, node1) not in selected_edges for n in selected_nodes) \
                and all((node2, n) not in selected_edges and (n, node2) not in selected_edges for n in selected_nodes):
            selected_edges.add(edge)
            selected_nodes.add(node1)
            selected_nodes.add(node2)

    # Count neighbors
    total_neighbor_count = 0
    counted_neighbors = set()

    for node in selected_nodes:
        for neighbor in range(len(adj_matrix[node])):
            if adj_matrix[node][neighbor] == 1 and neighbor not in selected_nodes and neighbor not in counted_neighbors:
                total_neighbor_count += 1
                counted_neighbors.add(neighbor)

    return total_neighbor_count


# 定义拟合函数 6n - 18
def fit_function(a, n, b):
    return a * n - b

import matplotlib.pyplot as plt
import os
from datetime import datetime


def draw(n, small_component_size, iteration_index, a, b):
    neighbors_counts_all = []

    # 生成迭代的 neighbors counts
    for i in range(iteration_index):
        matrix = generate_adjacency_matrix(n)
        #neighbors_counts = get_neighbors_count(matrix, small_component_size)
        neighbors_counts = get_edges_neighbors_count(matrix, small_component_size)
        neighbors_counts_all.append(neighbors_counts)
        print(neighbors_counts)

    # Convert the list of lists into a 2D NumPy array
    my_array = np.array(neighbors_counts_all)

    # Reshape the 1D array into a 2D array with shape (sqrt(length), sqrt(length))
    arr2d = my_array.reshape((int(np.sqrt(len(my_array))), int(np.sqrt(len(my_array)))))

    # Create meshgrid for X and Y
    X, Y = np.meshgrid(np.arange(arr2d.shape[1]), np.arange(arr2d.shape[0]))

    # Set up the figure and axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D wireframe
    ax.plot_wireframe(X, Y, arr2d)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Neighbors Counts')
    ax.set_title('Neighbors Counts in 3D Wireframe Plot')

    # Show the plot
    plt.show()
    #获取当前时间
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 构建文件名
    filename = f"plot_{current_time}.pdf"
    # 原始路径字符串
    windows_path = r'C:\Users\Administrator\Desktop\实验图\论文放'

    # 转换为Python绝对路径
    python_abs_path = os.path.abspath(windows_path)

    # 保存图形
    plt.savefig(os.path.join(python_abs_path, filename), dpi=600)
    # 显示图形
    plt.show()
def save(n, small_component_size, iteration_index, a, b):
    neighbors_counts_all = []

    # 生成迭代的 neighbors counts
    for i in range(iteration_index):
        matrix = generate_adjacency_matrix(n)
        neighbors_counts = get_edges_neighbors_count(matrix, small_component_size)
        neighbors_counts_all.append(neighbors_counts)
        print(neighbors_counts)

    # 将列表转换为NumPy数组
    my_array = np.array(neighbors_counts_all)
    #数组排序
    #my_array = np.sort(my_array, axis=None)

    # 将一维数组转换为二维数组，形状为(sqrt(length), sqrt(length))
    arr2d = my_array.reshape((int(np.sqrt(len(my_array))), int(np.sqrt(len(my_array)))))

    # 将数组保存为文本文件
    np.savetxt('data03.txt', arr2d, fmt='%d')


def drawSn():
    # 定义 n 和 small_component_size
    # 维度
    n = 5
    # 小分支大小
    small_component_size = 2
    # 邻居数量集合
    # 迭代次数
    iteration_index = 1600
    # an-b
    a = 4
    b = 10

    save(n, small_component_size, iteration_index, a, b)

#
drawSn()
# start_time = time.time()  # 记录开始时间
# for i in range(1, 100):
#     drawSn()
# end_time = time.time()  # 记录结束时间
# elapsed_time = end_time - start_time  # 计算经过的时间
# print("Time elapsed for draw function:", elapsed_time)
#
# drawSn()
def min_hamiltonian_cycle(adj_matrix):
    num_vertices = len(adj_matrix)
    # 构建所有可能的路径
    all_paths = itertools.permutations(range(num_vertices))

    min_length = float('inf')  # 设置一个初始的无穷大的长度
    min_cycle = None

    for path in all_paths:
        # 计算路径长度
        length = 0
        for i in range(num_vertices):
            length += adj_matrix[path[i]][path[(i + 1) % num_vertices]]  # 对路径中相邻顶点的权重进行求和

        # 更新最小长度和对应的路径
        if length < min_length:
            min_length = length
            min_cycle = path

    return min_length

def graph_diameter(adj_matrix):
    num_vertices = len(adj_matrix)
    dist_matrix = adj_matrix  # 对于无权重图，邻接矩阵本身就是距离矩阵

    # 使用 Floyd-Warshall 算法计算任意两点间的最短路径长度
    for k in range(num_vertices):
        for i in range(num_vertices):
            for j in range(num_vertices):
                dist_matrix[i][j] = min(dist_matrix[i][j], dist_matrix[i][k] + dist_matrix[k][j])

    # 寻找距离矩阵中的最大值，即图的直径
    diameter = max(max(row) for row in dist_matrix)
    return diameter



