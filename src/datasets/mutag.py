# https://github.com/flyingdoog/PGExplainer/blob/master/MUTAG.ipynb

import yaml
import torch
import numpy as np
import pickle as pkl
from pathlib import Path
from torch_geometric.data import InMemoryDataset, Data


class Mutag(InMemoryDataset):
    def __init__(self, root):
        super().__init__(root=root)
        self.process() #MANGO
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['Mutagenicity_A.txt', 'Mutagenicity_edge_gt.txt', 'Mutagenicity_edge_labels.txt',
                'Mutagenicity_graph_indicator.txt', 'Mutagenicity_graph_labels.txt', 'Mutagenicity_label_readme.txt',
                'Mutagenicity_node_labels.txt', 'Mutagenicity.pkl']

    @property
    def processed_file_names(self):
        return ['dual_data.pt'] #MANGO

    def download(self):
        raise NotImplementedError

    def process(self):
        with open(self.raw_dir + '/Mutagenicity.pkl', 'rb') as fin:
            _, original_features, original_labels = pkl.load(fin)

        dual_edge_lists, graph_labels, dual_node_gt_lists, dual_node_label_lists, dual_node_lists = self.get_graph_data()

        #print("\U0001F96D original_features shape:", original_features.shape) #MANGO
        # print("\U0001F96D original_labels shape:", original_labels.shape) #MANGO
        # print("\U0001F96D edge_lists len:", len(dual_edge_lists)) #MANGO
        # print("\U0001F96D graph_labels shape:", graph_labels.shape) #MANGO
        # print("\U0001F96D dual_node_gt_list len:", len(dual_node_gt_lists)) #MANGO
        # print("\U0001F96D node_type_lists len:", len(dual_node_label_lists)) #MANGO
        # print("\U0001F96D dual_nodes_list len:", len(dual_node_lists)) #MANGO

        # print("\U0001F96D original_features first rows", original_features[:,0,:]) #MANGO
        # print("\U0001F96D original_features last rows", original_features[:,417,:]) #MANGO
        # print("\U0001F96D original_features last rows any non zero", np.any(original_features[:,417,:] != 0)) #MANGO

        #print("\U0001F96D node_type_lists:", dual_node_label_lists) #MANGO

        data_list = []

        #print("\U0001F96D dual_edge_list:", dual_edge_lists) #MANGO
        #print("\U0001F96D dual_node_lists:", dual_node_lists) #MANGO
        count = 0
        for i in range(original_labels.shape[0]):
            num_nodes = len(dual_node_lists[i]) # MANGO
            print("\U0001F96D i:", i)
            print("\U0001F96D num_nodes:", num_nodes)
            edge_index = torch.tensor(dual_edge_lists[i], dtype=torch.long).T # MANGO
            dual_nodes_index = torch.tensor(dual_node_lists[i], dtype=torch.long).T

            y = torch.tensor(graph_labels[i]).float().reshape(-1, 1)
            x = torch.tensor(original_features[i][:num_nodes]).float() # MANGO removes padding, MANGO PIT
            print("\U0001F96D padding:", original_features[i][num_nodes:])
            print("\U0001F96D padding sum:", original_features[i][num_nodes:].sum())
            #assert original_features[i][num_nodes:].sum() == 0
            #if (original_features[i][num_nodes:].sum() != 0):
            #    count+=1

            # MANGO
            dual_node_gt = torch.tensor(dual_node_gt_lists[i]).float() # MANGO 
            #if y.item() != 0: # MANGO
            #    edge_label = torch.zeros_like(edge_label).float() # MANGO zero out edge labels, RAW MANGO

            node_label = torch.zeros(x.shape[0]) # MANGO copies shape of node features and sets to zero, RAW MANGO
            #signal_nodes = list(set(dual_nodes_index[:, dual_node_signal.bool()].reshape(-1).tolist())) # MANGO picks out signal nodes from the edge label (for explainer; ground truth nodes), ONLY THE RIPE MANGO
            #print("\U0001F96D dual_nodes_index:", dual_nodes_index)
            #print("\U0001F96D edge_index:", edge_index)
            
            #signal_nodes = list(set(dual_nodes_index[dual_node_gt.astype(bool)].tolist()))
            signal_nodes = list(set([x - 1 for x in dual_nodes_index[:, dual_node_gt.bool()].reshape(-1).tolist()]))
            #signal_nodes = list(dual_nodes[dual_node_signal])

            #print("\U0001F96D dual_gt_list:", dual_node_gt_lists) #MANGO

            if y.item() == 0:
                print("\U0001F96D signal_nodes:", signal_nodes)
                node_label[signal_nodes] = 1 #set signal nodes' label to one if graph label is 0, ONLY THE RIPE MANGO
                #print("\U0001F96D signal_nodes:", signal_nodes)

            # if len(signal_nodes) != 0: #MANGO IDK HERE DOES IT MAKE SENSE TO TAKE THIS OUT cause {4,1} {4,3} etc doesn't apply?*, ROTTEN MANGO?
            #     node_type = torch.tensor(dual_node_label_lists[i]) # MANGO
            #     node_type = set(node_type[signal_nodes].tolist())
            #     assert node_type in ({4, 1}, {4, 3}, {4, 1, 3})  # MANGO NO or NH; Checks that signal nodes are the right node type

            # if y.item() == 0 and len(signal_nodes) == 0: #SKIP THIS 
            #     continue

            edge_label = torch.zeros(edge_index.size(1), dtype=torch.float) # MANGO 

            data_list.append(Data(x=x, y=y, edge_index=edge_index, node_label=node_label, edge_label=edge_label, node_type=torch.tensor(dual_node_label_lists[i]))) # MANGO 

        print("\U0001F96D count:", count)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_graph_data(self):
        pri = self.raw_dir + '/Mutagenicity_'

        file_edges = pri + 'A.txt' 
        file_edge_labels = pri + 'edge_labels.txt'
        file_edge_gt = pri + 'edge_gt.txt'
        file_graph_indicator = pri + 'graph_indicator.txt'
        file_graph_labels = pri + 'graph_labels.txt'
        file_node_labels = pri + 'node_labels.txt'

        dual_nodes = np.loadtxt(file_edges, delimiter=',').astype(np.int32) # MANGO loading dual_nodes in adjacency matrix form, SLICED MANGO
        try:
            dual_node_labels = np.loadtxt(file_edge_labels, delimiter=',').astype(np.int32)
        except Exception as e:
            print(e)
            print('use edge label 0')
            dual_node_labels = np.zeros(dual_nodes.shape[0]).astype(np.int32) # MANGO

        try: # MANGO
            dual_node_signal = np.loadtxt(file_edge_gt, delimiter=',').astype(np.int32) #MANGO
        except Exception as e: # MANGO
            print(e) # MANGO
            print('use edge gt 0') # MANGO
            dual_node_signal = np.zeros(dual_nodes.shape[0]).astype(np.int32) # MANGO


        graph_indicator = np.loadtxt(file_graph_indicator, delimiter=',').astype(np.int32)
        graph_labels = np.loadtxt(file_graph_labels, delimiter=',').astype(np.int32)

        dual_graph_indicator = np.zeros((dual_nodes.shape[0], 1)) #MANGO, 
        for i in (range(dual_nodes.shape[0])): #MANGO
            assert graph_indicator[(dual_nodes[i][0]-1)] == graph_indicator[(dual_nodes[i][1]-1)]
            dual_graph_indicator[i] = graph_indicator[(dual_nodes[i][0])-1] #MANGO
        dual_graph_indicator = dual_graph_indicator.astype(int) # MANGO

        try:
            primal_node_labels = np.loadtxt(file_node_labels, delimiter=',').astype(np.int32) # MANGO should these be included in the edge features? MULTIPLE MANGO
        except Exception as e:
            print(e)
            print('use node label 0')
            primal_node_labels = np.zeros(graph_indicator.shape[0]).astype(np.int32) # MANGO

        # ATP loaded all files whole MANGO has been eaten

        dual_edges = [] # MANGO

        #CHATGPT ALERT ALERT ALERT MANGOS FALLING
        from collections import defaultdict

        dual_nodes = np.array(dual_nodes)  # Just in case

        group_by_first = defaultdict(list)
        group_by_second = defaultdict(list)

        for idx, (a, b) in enumerate(dual_nodes):
            group_by_first[a].append(idx)
            group_by_second[b].append(idx)

        dual_edges = []

        def add_pairs_from_group(group):
            indices = group
            length = len(indices)
            for i in range(length):
                for j in range(i + 1, length):
                    e1 = dual_nodes[indices[i]]
                    e2 = dual_nodes[indices[j]]
                    #print("\U0001F96D ENDDDDDD1:", e1[0], e2[0], e1[1], e2[1])
                    dual_edges.append([e1.tolist(), e2.tolist()])

        # Add pairs grouped by first element
        for group in group_by_first.values():
            if len(group) > 1:
                add_pairs_from_group(group)

        # Add pairs grouped by second element
        for group in group_by_second.values():
            if len(group) > 1:
                add_pairs_from_group(group)

        # CHATGPT END THE MANGOS HAVE LANDED

        dual_edges = np.array(dual_edges) # MANGO

        graph_id = 1
        starts = [1]
        node2graph = {}
        #print("\U0001F96D dual_node_signal:", dual_graph_indicator) #MANGO
        for i in range(len(dual_graph_indicator)): #MANGO
            if dual_graph_indicator[i] != graph_id: #MANGO 
                graph_id = dual_graph_indicator[i] #MANGO
                starts.append(i) #MANGO
            #node2graph[tuple(dual_nodes[i])] = len(starts)-1 #MANGO

        graph_id = 1
        primal_starts = [1]
        for i in range(len(graph_indicator)): #MANGO
            if graph_indicator[i] != graph_id: #MANGO 
                graph_id = graph_indicator[i] #MANGO
                primal_starts.append(i+1) #MANGO
            node2graph[i+1] = len(primal_starts)-1 #MANGO

        #print("\U0001F96D dual graph indicator:", dual_graph_indicator) #MANGO

        # print(starts)
        # print(node2graph)
        graphid = 0
        dual_edge_lists = [] # MANGO
        #edge_label_lists = [] # MANGO
        dual_edge_list = [] # MANGO
        #edge_label_list = [] # MANGO
        for (s, t) in list(dual_edges): #MANGO removed edge labels because dual edges have no gt, SEEDLESS MANGO 
            sgid = node2graph[s[0]] # MANGO
            tgid = node2graph[t[0]] # MANGO
            if sgid != tgid:
                print('edges connecting different graphs, error here, please check.')
                print(s, t, 'graph id', sgid, tgid)
                exit(1)
            gid = sgid
            if gid != graphid:
                dual_edge_lists.append(dual_edge_list) # MANGO
                #edge_label_lists.append(edge_label_list) # MANGO
                dual_edge_list = []
                #edge_label_list = [] # MANGO
                graphid = gid
            start = starts[gid] # MANGO
            dual_edge_list.append((s-(start+1)/2, t-(start+1)/2))
            #edge_label_list.append(l) # MANGO

        dual_edge_lists.append(dual_edge_list)
        #edge_label_lists.append(edge_label_list) # MANGO

        #CHATGPT ALERT ALERT ALERT MANGOS FALLING

        dual_node_gt_lists = []
        dual_node_gt_list = []
        graphid = 0

        #print("\U0001F96D dual_node_signal:", dual_node_signal) #MANGO


        for idx, label in enumerate(dual_node_signal):
            gid = node2graph[dual_nodes[idx][0]]  # graph ID for this node
    
            if gid != graphid:
                dual_node_gt_lists.append(dual_node_gt_list)
                dual_node_gt_list = []
                graphid = gid

            dual_node_gt_list.append(label)

        dual_node_gt_lists.append(dual_node_gt_list)

        #CHATGPT END THE MANGOS HAVE LANDED 

        
        graphid = 0
        

        # for i in range(len(dual_nodes)):
        #     gid = node2graph[tuple(dual_nodes[i])]
    
        #     if gid != graphid:
        #         dual_nodes_lists.append(dual_node_list)
        #         dual_node_list = []
        #         graphid = gid

        #     start = starts[gid]
        #     dual_node_list.append(tuple(dual_nodes[i] - start))  # or just dual_nodes[i] if raw
    
        # dual_nodes_lists.append(dual_node_list)

        # node labels


        # print("\U0001F96D dual_nodes size:", len(dual_nodes)) #MANGO
        dual_node_lists = []
        dual_node_label_lists = []
        graphid = 0
        dual_node_label_list = []
        dual_node_list = []
        for i in range(len(dual_node_labels)):
            # nid = i+1 # MANGO
            gid = node2graph[dual_nodes[i][0]] # MANGO
            # start = starts[gid]
            if gid != graphid:
                dual_node_label_lists.append(dual_node_label_list)
                dual_node_lists.append(dual_node_list)
                graphid = gid
                dual_node_label_list = []
                dual_node_list = []
            u, v = dual_nodes[i]
            gu = graph_indicator[u-1]
            gv = graph_indicator[v-1]
            assert gu == gv

            start = primal_starts[gid]
            print("\U0001F96D start:", start) #MANGO
            print("\U0001F96D dual_nodes[i]:", dual_nodes[i])
            dual_node_label_list.append(dual_node_labels[i])
            dual_node_list.append((dual_nodes[i][0]-start, dual_nodes[i][1]-start))
        dual_node_label_lists.append(dual_node_label_list)
        dual_node_lists.append(dual_node_list)

        # print("\U0001F96D dual_node_label_lists size:", len(dual_node_label_lists)) #MANGO
        # print("\U0001F96D dual_node_lists size:", len(dual_node_lists)) #MANGO

        return dual_edge_lists, graph_labels, dual_node_gt_lists, dual_node_label_lists, dual_node_lists # MANGO


# MANGO OG CODE MANGO

# # https://github.com/flyingdoog/PGExplainer/blob/master/MUTAG.ipynb

# import yaml
# import torch
# import numpy as np
# import pickle as pkl
# from pathlib import Path
# from torch_geometric.data import InMemoryDataset, Data


# class Mutag(InMemoryDataset):
#     def __init__(self, root):
#         super().__init__(root=root)
#         self.process() #MANGO
#         self.data, self.slices = torch.load(self.processed_paths[0])

#     @property
#     def raw_file_names(self):
#         return ['Mutagenicity_A.txt', 'Mutagenicity_edge_gt.txt', 'Mutagenicity_edge_labels.txt',
#                 'Mutagenicity_graph_indicator.txt', 'Mutagenicity_graph_labels.txt', 'Mutagenicity_label_readme.txt',
#                 'Mutagenicity_node_labels.txt', 'Mutagenicity.pkl']

#     @property
#     def processed_file_names(self):
#         return ['data.pt']

#     def download(self):
#         raise NotImplementedError

#     def process(self):
#         with open(self.raw_dir + '/Mutagenicity.pkl', 'rb') as fin:
#             _, original_features, original_labels = pkl.load(fin)

#         edge_lists, graph_labels, edge_label_lists, node_type_lists = self.get_graph_data()

#         print("\U0001F96D original_features shape:", original_features.shape) #MANGO
#         print("\U0001F96D original_labels shape:", original_labels.shape) #MANGO
#         print("\U0001F96D edge_lists len:", len(edge_lists)) #MANGO
#         print("\U0001F96D graph_labels shape:", graph_labels.shape) #MANGO
#         print("\U0001F96D edge_label_lists len:", len(edge_label_lists)) #MANGO
#         print("\U0001F96D node_type_lists len:", len(node_type_lists)) #MANGO

#         print("\U0001F96D original_features first rows", original_features[:,0,:]) #MANGO
#         print("\U0001F96D original_features last rows", original_features[:,417,:]) #MANGO
#         print("\U0001F96D original_features last rows any non zero", np.any(original_features[:,417,:] != 0)) #MANGO

#         #print("\U0001F96D node_type_lists:", node_type_lists) #MANGO

#         data_list = []

#         print("\U0001F96D dual_edge_list:", edge_lists) #MANGO

#         data_list = []
#         for i in range(original_labels.shape[0]):
#             num_nodes = len(node_type_lists[i])
#             edge_index = torch.tensor(edge_lists[i], dtype=torch.long).T

#             y = torch.tensor(graph_labels[i]).float().reshape(-1, 1)
#             x = torch.tensor(original_features[i][:num_nodes]).float()
#             assert original_features[i][num_nodes:].sum() == 0
#             edge_label = torch.tensor(edge_label_lists[i]).float()
#             if y.item() != 0:
#                 edge_label = torch.zeros_like(edge_label).float()

#             node_label = torch.zeros(x.shape[0])
#             signal_nodes = list(set(edge_index[:, edge_label.bool()].reshape(-1).tolist()))

#             print("\U0001F96D signal_nodes:", signal_nodes) #MANGO
#             print("\U0001F96D edge_label:", edge_label) #MANGO
#             #print("\U0001F96D edge_labe bool:", edge_label.bool())
#             print("\U0001F96D edge_index:", edge_index ) #MANGO
#             print("\U0001F96D edge_lists[i]:", edge_lists[i] ) #MANGO
#             # print("\U0001F96D signal_nodes len:", len(signal_nodes)) #MANGO
#             # non_empty_count = sum(len(nodes) > 0 for nodes in signal_nodes)
#             # print("\U0001F96D Non-empty signal_nodes count:", non_empty_count) #MANGO
            
#             if y.item() == 0:
#                 node_label[signal_nodes] = 1

#             if len(signal_nodes) != 0:
#                 node_type = torch.tensor(node_type_lists[i])
#                 node_type = set(node_type[signal_nodes].tolist())
#                 assert node_type in ({4, 1}, {4, 3}, {4, 1, 3})  # NO or NH

#             if y.item() == 0 and len(signal_nodes) == 0:
#                 continue

#             data_list.append(Data(x=x, y=y, edge_index=edge_index, node_label=node_label, edge_label=edge_label, node_type=torch.tensor(node_type_lists[i])))

#         data, slices = self.collate(data_list)
#         torch.save((data, slices), self.processed_paths[0])

#     def get_graph_data(self):
#         pri = self.raw_dir + '/Mutagenicity_'

#         file_edges = pri + 'A.txt'
#         # file_edge_labels = pri + 'edge_labels.txt'
#         file_edge_labels = pri + 'edge_gt.txt'
#         file_graph_indicator = pri + 'graph_indicator.txt'
#         file_graph_labels = pri + 'graph_labels.txt'
#         file_node_labels = pri + 'node_labels.txt'

#         edges = np.loadtxt(file_edges, delimiter=',').astype(np.int32)
#         try:
#             edge_labels = np.loadtxt(file_edge_labels, delimiter=',').astype(np.int32)
#         except Exception as e:
#             print(e)
#             print('use edge label 0')
#             edge_labels = np.zeros(edges.shape[0]).astype(np.int32)

#         graph_indicator = np.loadtxt(file_graph_indicator, delimiter=',').astype(np.int32)
#         graph_labels = np.loadtxt(file_graph_labels, delimiter=',').astype(np.int32)

#         try:
#             node_labels = np.loadtxt(file_node_labels, delimiter=',').astype(np.int32)
#         except Exception as e:
#             print(e)
#             print('use node label 0')
#             node_labels = np.zeros(graph_indicator.shape[0]).astype(np.int32)

#         graph_id = 1
#         starts = [1]
#         node2graph = {}
#         for i in range(len(graph_indicator)):
#             if graph_indicator[i] != graph_id:
#                 graph_id = graph_indicator[i]
#                 starts.append(i+1)
#             node2graph[i+1] = len(starts)-1
#         # print(starts)
#         # print(node2graph)
#         graphid = 0
#         edge_lists = []
#         edge_label_lists = []
#         edge_list = []
#         edge_label_list = []
#         for (s, t), l in list(zip(edges, edge_labels)):
#             sgid = node2graph[s]
#             tgid = node2graph[t]
#             if sgid != tgid:
#                 print('edges connecting different graphs, error here, please check.')
#                 print(s, t, 'graph id', sgid, tgid)
#                 exit(1)
#             gid = sgid
#             if gid != graphid:
#                 edge_lists.append(edge_list)
#                 edge_label_lists.append(edge_label_list)
#                 edge_list = []
#                 edge_label_list = []
#                 graphid = gid
#             start = starts[gid]
#             edge_list.append((s-start, t-start))
#             edge_label_list.append(l)

#         edge_lists.append(edge_list)
#         edge_label_lists.append(edge_label_list)

#         # node labels
#         node_label_lists = []
#         graphid = 0
#         node_label_list = []
#         for i in range(len(node_labels)):
#             nid = i+1
#             gid = node2graph[nid]
#             # start = starts[gid]
#             if gid != graphid:
#                 node_label_lists.append(node_label_list)
#                 graphid = gid
#                 node_label_list = []
#             node_label_list.append(node_labels[i])
#         node_label_lists.append(node_label_list)

#         return edge_lists, graph_labels, edge_label_lists, node_label_lists