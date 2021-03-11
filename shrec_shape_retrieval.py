import pickle
from grakel import GraphKernel
import os.path as osp
import numpy as np
from numpy import linalg as LA
from grakel import Graph
from grakel.kernels import PropagationAttr
from grakel.kernels import WeisfeilerLehman, VertexHistogram, EdgeHistogram, RandomWalk, GraphHopper, PyramidMatch

def feature_vector(p,adj,m,t):
    feature_vec = np.zeros((t,m))
    l=0.75
    for i_step in range(t):
        p = np.matmul(adj, p)
        temp = p.reshape(1,-1)
        feature_vec[i_step] = temp[0,:m]*np.power(l,i_step)
    f = np.sum(feature_vec,0)/t
    return f
    
def weighted_dist_metric(vec1, vec2):
    l = 0.75
    dist = 0
    for i in range(len(vec1)):
        dist = dist + np.power(l,i)*np.abs(vec1[i]-vec2[i])
    return dist
 
def generate_single_neighbor_subgraph(vi, G, K):
    # produce neighbor subgraph
    N = G.produce_neighborhoods(r=K,purpose='dictionary')
    g = G.get_subgraph(N[K][vi])
    return g  

num_sphere = 512
num_graph = 300

#data_dir = osp.join('real_adj_nor.txt') 
data_dir = osp.join('synthetic_adj_nor.txt') 
fp = open(data_dir,'rb')
dataset = pickle.load(fp)

# # sub graph
# sub_graph=[0]*num_graph
# for i_graph in range(num_graph):
    # sub_graph[i_graph] = generate_single_neighbor_subgraph(0, dataset['x'][i_graph], 2)
    # print(i_graph)
# fp2 = open('synthetic_K_2.txt', 'wb')
# pickle.dump(sub_graph, fp2)

# data_dir2 = osp.join('synthetic_K_2.txt')
# fp2 = open(data_dir2,'rb')
# sub_graph = pickle.load(fp2)

subject = dataset['x']

# for i in range(num_graph):
    # print(sub_graph[i].n)

#sp_kernel = GraphKernel(kernel="shortest_path", normalize=True)
#kernel = GraphKernel(kernel="neighborhood_hash", normalize=True)
#kernel = GraphKernel(kernel="graph_hopper", normalize=True)
#kernel = GraphKernel(kernel="pyramid_match", normalize=True)
#kernel = GraphKernel(kernel="propagation", normalize=True)
# kernel = GraphKernel(kernel = [{"name": "weisfeiler_lehman", "n_iter": 5}, "subtree_wl"], normalize=True)
# kernel = GraphKernel(kernel = [{"name": "weisfeiler_lehman", "n_iter": 5}, "pyramid_match"], normalize=True)
#rw_kernel = GraphKernel(kernel="random_walk", normalize=True)
#kernel= GraphKernel(kernel=[{"name": "random_walk","with_labels":True}], normalize=True)
#wl = WeisfeilerLehman(base_graph_kernel=VertexHistogram, normalize=True)
#wl = WeisfeilerLehman(base_graph_kernel=EdgeHistogram, normalize=True)
#wl = WeisfeilerLehman(base_graph_kernel=PyramidMatch, normalize=True)

# # pre-process
# for i_graph in range(num_graph):
    # for i_ver in range(subject[i_graph].n):
        # subject[i_graph].index_node_labels[i_ver] = int(subject[i_graph].index_node_labels[i_ver]*25)/25

# rank_mat = np.zeros((num_graph,num_graph-1))
# kernel = GraphKernel(kernel = [{"name": "weisfeiler_lehman", "n_iter": 5}, "subtree_wl"], normalize=True)
# dist_mat = kernel.fit_transform(subject)
# print(dist_mat[:,:10])

# for i_graph in range(num_graph):
    # # index sort
    # ranked = np.argsort(dist_mat[i_graph])
    # largest_indices = ranked[::-1]
    # print(largest_indices[:10])
    # rank_mat[i_graph] = largest_indices[1:]


n=20
m=20
t=3
rank_mat = np.zeros((num_graph,num_graph-1))
dist_mat = np.zeros((num_graph,num_graph))
for i_graph in range(num_graph):
    num_sphere_i = subject[i_graph].n
    ri_list = np.array([subject[i_graph].index_node_labels[i_ver] for i_ver in range(num_sphere_i)])
    ri_list = ri_list.reshape(num_sphere_i,1)
    adj_i = subject[i_graph].adjacency_matrix
    f_i = feature_vector(ri_list,adj_i,m,t)
    for j_graph in range(i_graph,num_graph):
        num_sphere_j = subject[j_graph].n
        rj_list = np.array([subject[j_graph].index_node_labels[i_ver] for i_ver in range(num_sphere_j)])
        rj_list = rj_list.reshape(num_sphere_j,1)
        adj_j = subject[j_graph].adjacency_matrix
        f_j = feature_vector(rj_list,adj_j,m,t)
        w=1
        dist1 = weighted_dist_metric(ri_list[:n], rj_list[:n])
        dist2 = weighted_dist_metric(f_i, f_j)
        dist_mat[i_graph, j_graph] = dist1 + w*dist2
        #dist_mat[i_graph, j_graph] = LA.norm(ri_list[:n]-rj_list[:n]) + w*LA.norm(f_i-f_j)
        dist_mat[j_graph, i_graph] = dist_mat[i_graph, j_graph]
    ranked = np.argsort(dist_mat[i_graph])
    print(ranked[:10])      
    rank_mat[i_graph] = ranked[1:]

#np.savetxt('real_T1_whole.txt', rank_mat, delimiter=' ')
np.savetxt('synthetic_T1_whole.txt', rank_mat, delimiter=' ')
#np.savetxt('synthetic_shrec_T1_dataset_256.txt', rank_mat, delimiter=' ')



