import numpy as np
import os.path as osp
import pickle
from scipy.sparse import csr_matrix
from grakel import Graph
from spheres2graph_kernel import * 

def produce_graphs(dir_path,num_sphere):
    resolution=0.05
    num_graph = 300
    graphs = [0]*num_graph
    
    normalize_path = "./SHREC_14/Synthetic_128/normalization_data.txt"
    nor_data = np.loadtxt(normalize_path, delimiter=' ')
    
    i_data=0
    for i_file in range(num_graph):
        file_path = osp.join(dir_path, str(i_file)+".txt")
        edge_index, node_features, edge_attr, pseudo, zero_adj = spheres2graph(file_path, num_sphere, resolution, normalize_radii=False)
        truncated_num = len(node_features)
        node_features = node_features.squeeze().numpy().tolist()
        edge_attr = edge_attr.squeeze().numpy().tolist()
        vs = edge_index[0,:].squeeze().numpy().tolist()
        vt = edge_index[1,:].squeeze().numpy().tolist()
        adj = csr_matrix((edge_attr, (vs,vt) ),shape=(truncated_num,truncated_num))
        node_attr = {i : node_features[i][3]/nor_data[i_file, 3] for i in range(truncated_num)}
        graphs[i_file] = Graph(adj,node_labels=node_attr)
        print(i_data)
        i_data += 1
    dataset = {
        'x': graphs
    }  
    return dataset

dir_path = "./SHREC_14/Synthetic_128/"


num_sphere = 512
dataset_shrec = produce_graphs(dir_path,num_sphere)

fp = open('synthetic_adj_nor.txt', 'wb')
pickle.dump(dataset_shrec, fp)



