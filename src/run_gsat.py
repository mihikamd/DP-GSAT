import yaml
import scipy
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_sparse import transpose
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph, is_undirected
from ogb.graphproppred import Evaluator
from sklearn.metrics import roc_auc_score
from rdkit import Chem

import torch.nn.functional as F

from utils import NSALoss, LNSA_loss
from sklearn.manifold import TSNE

#from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from pretrain_clf import train_clf_one_seed
from utils import Writer, Criterion, MLP, visualize_a_graph, save_checkpoint, load_checkpoint, get_preds, get_lr, set_seed, process_data
from utils import get_local_config_name, get_model, get_data_loaders, write_stat_from_metric_dicts, reorder_like, init_metric_dict


class GSAT(nn.Module):
    #treat as two separate models running inside of gsat and just duplicate everything? 
    def __init__(self, primal_clf, primal_extractor, primal_optimizer, primal_scheduler, primal_writer, primal_device, primal_model_dir, primal_dataset_name, primal_num_class, primal_multi_label, primal_random_state,
                 primal_method_config, primal_shared_config, primal_model_config, dual_clf, dual_extractor, dual_optimizer, dual_scheduler, dual_writer, dual_device, dual_model_dir, dual_dataset_name, dual_num_class, dual_multi_label, dual_random_state,
                 dual_method_config, dual_shared_config, dual_model_config):
        super().__init__()
        # self.alpha = alpha
        # self.lamb = lamb

        self.primal_clf = primal_clf
        self.primal_extractor = primal_extractor
        self.primal_optimizer = primal_optimizer
        self.primal_scheduler = primal_scheduler

        self.primal_writer = primal_writer
        self.primal_device = primal_device
        self.primal_model_dir = primal_model_dir
        self.primal_dataset_name = primal_dataset_name
        self.primal_random_state = primal_random_state
        self.primal_method_name = primal_method_config['method_name']

        self.primal_learn_edge_att = primal_shared_config['learn_edge_att']
        self.primal_k = primal_shared_config['precision_k']
        self.primal_num_viz_samples = primal_shared_config['num_viz_samples']
        self.primal_viz_interval = primal_shared_config['viz_interval']
        self.primal_viz_norm_att = primal_shared_config['viz_norm_att']

        self.primal_epochs = primal_method_config['epochs']
        self.primal_pred_loss_coef = primal_method_config['pred_loss_coef']
        self.primal_info_loss_coef = primal_method_config['info_loss_coef']

        self.primal_fix_r = primal_method_config.get('fix_r', None)
        self.primal_decay_interval = primal_method_config.get('decay_interval', None)
        self.primal_decay_r = primal_method_config.get('decay_r', None)
        self.primal_final_r = primal_method_config.get('final_r', 0.1)
        self.primal_init_r = primal_method_config.get('init_r', 0.9)

        self.primal_multi_label = primal_multi_label
        self.primal_criterion = Criterion(primal_num_class, primal_multi_label)

        self.primal_hidden_size = primal_model_config['hidden_size']

        self.primal_mask_mlp = nn.Sequential(
            nn.Linear(self.primal_hidden_size, self.primal_hidden_size),
            nn.ReLU(),
            nn.Linear(self.primal_hidden_size, 2)
        )

        #DUAL
        self.dual_clf = dual_clf
        self.dual_extractor = dual_extractor
        self.dual_optimizer = dual_optimizer
        self.dual_scheduler = dual_scheduler

        self.dual_writer = dual_writer
        self.dual_device = dual_device
        self.dual_model_dir = dual_model_dir
        self.dual_dataset_name = dual_dataset_name
        self.dual_random_state = dual_random_state
        self.dual_method_name = dual_method_config['method_name']

        self.dual_learn_edge_att = dual_shared_config['learn_edge_att']
        self.dual_k = dual_shared_config['precision_k']
        self.dual_num_viz_samples = dual_shared_config['num_viz_samples']
        self.dual_viz_interval = dual_shared_config['viz_interval']
        self.dual_viz_norm_att = dual_shared_config['viz_norm_att']

        self.dual_epochs = dual_method_config['epochs']
        self.dual_pred_loss_coef = dual_method_config['pred_loss_coef']
        self.dual_info_loss_coef = dual_method_config['info_loss_coef']

        self.dual_fix_r = dual_method_config.get('fix_r', None)
        self.dual_decay_interval = dual_method_config.get('decay_interval', None)
        self.dual_decay_r = dual_method_config.get('decay_r', None)
        self.dual_final_r = dual_method_config.get('final_r', 0.1)
        self.dual_init_r = dual_method_config.get('init_r', 0.9)

        self.dual_multi_label = dual_multi_label
        self.dual_criterion = Criterion(dual_num_class, dual_multi_label)

        self.dual_hidden_size = dual_model_config['hidden_size']

        self.dual_mask_mlp = nn.Sequential(
            nn.Linear(self.dual_hidden_size, self.dual_hidden_size),
            nn.ReLU(),
            nn.Linear(self.dual_hidden_size, 2)
        )

    def __loss__(self, primal_att, dual_att, primal_clf_logits, dual_clf_logits, primal_clf_labels, dual_clf_labels, dual_att_log_logits, epoch): #just used primal self. for all of this
        primal_pred_loss = self.primal_criterion(primal_clf_logits, primal_clf_labels)

        dual_pred_loss = self.dual_criterion(dual_clf_logits, dual_clf_labels)

        dual_r = self.dual_fix_r if self.dual_fix_r else self.get_r(self.dual_decay_interval, self.dual_decay_r, epoch, final_r=self.dual_final_r, init_r=self.dual_init_r)
        dual_info_loss = (dual_att * torch.log(dual_att/dual_r + 1e-6) + (1-dual_att) * torch.log((1-dual_att)/(1-dual_r+1e-6) + 1e-6)).mean()

        primal_r = dual_att_log_logits.sigmoid().detach()

        #primal_r = self.primal_fix_r if self.primal_fix_r else self.get_r(self.primal_decay_interval, self.primal_decay_r, epoch, final_r=self.primal_final_r, init_r=self.primal_init_r)
        primal_info_loss = (primal_att * torch.log(primal_att/primal_r + 1e-6) + (1-primal_att) * torch.log((1-primal_att)/(1-primal_r+1e-6) + 1e-6)).mean()

        primal_pred_loss = primal_pred_loss * self.primal_pred_loss_coef
        primal_info_loss = primal_info_loss * self.primal_info_loss_coef

        dual_pred_loss = dual_pred_loss * self.dual_pred_loss_coef
        dual_info_loss = dual_info_loss * self.dual_info_loss_coef

        # print(f"primal_info_loss: {primal_info_loss}, primal_pred_loss: {primal_pred_loss}, dual_info_loss: {dual_info_loss}, dual_pred_loss: {dual_pred_loss}")

        loss = primal_pred_loss + dual_pred_loss + primal_info_loss + dual_info_loss
        primal_loss_dict = {'loss': loss.item(), 'pred': primal_pred_loss.item(), 'info': primal_info_loss.item()}
        dual_loss_dict = {'loss': loss.item(), 'pred': dual_pred_loss.item(), 'info': dual_info_loss.item()}

        loss_dict = primal_loss_dict.copy() 
        loss_dict.update(dual_loss_dict)

        return loss, loss_dict
    
    def f1_sparsity_loss(self, p_uv, y_uv, eps=1e-6):
        TP = TP = (p_uv.view(-1) * y_uv.view(-1)).sum()


        # print(f"p_uv.shape: {p_uv.shape}, y_uv.shape: {y_uv.shape}")
        # print("p_uv: ", p_uv)
        # print("y_uv: ", y_uv)
        # print("p_uv * y_uv: ", (p_uv.view(-1) * y_uv.view(-1)))
        P = p_uv.sum()

        G = y_uv.sum()

        precision = TP / (P + eps)
        recall = TP / (G + eps)

        assert (p_uv >= 0).all() and (p_uv <= 1).all()
        assert (y_uv >= 0).all() and (y_uv <= 1).all()
        if (y_uv == 0).all():
            input("all zero")

        f1 = 2 * precision * recall / (precision + recall + eps)

        l1_loss = p_uv.abs().mean()

        # Total loss
        total_loss = (1 - f1) + l1_loss

        #print(f"TP={TP.item()}, P={P.item()}, G={G.item()}, precision={precision.item()}, recall={recall.item()}, f1={f1.item()}")

        return total_loss
    
    def gumbel_sigmoid(self, logits, tau=1.0, eps=1e-10):
        """Differentiable binary Gumbel-sigmoid sampling."""
        U = torch.rand_like(logits)
        g = -torch.log(-torch.log(U + eps) + eps)
        y = torch.sigmoid((logits + g) / tau)
        return y
        
    def dual_forward_pass(self, primal_data, dual_data, epoch, training):
        # Primal 
        primal_emb = self.primal_clf.get_emb(primal_data.x, primal_data.edge_index, batch=primal_data.batch, edge_attr=primal_data.edge_attr)

        #primal_mask_logits = self.primal_mask_mlp(primal_emb)  # [n_nodes, 2]
        #primal_node_mask = F.gumbel_softmax(primal_mask_logits, tau=1.0, hard=True, dim=-1)[:, 0]  # [n_nodes]

        # Apply node mask
        #primal_masked_emb = primal_emb * primal_node_mask.unsqueeze(-1)

        primal_att_log_logits = self.primal_extractor(primal_emb, primal_data.edge_index, primal_data.batch, "primal")


        #print("primal_att_log_logits:", primal_att_log_logits[:10])

        primal_node_att = self.sampling(primal_att_log_logits, epoch, training)


        # Dual 
        dual_emb = self.dual_clf.get_emb(dual_data.x, dual_data.edge_index, batch=dual_data.batch, edge_attr=dual_data.edge_attr)
        dual_att_log_logits = self.dual_extractor(dual_emb, dual_data.edge_index, dual_data.batch, "dual")


        # min_val = dual_att_log_logits.min()
        # max_val = dual_att_log_logits.max()

        # dual_att_log_logits = (dual_att_log_logits - min_val) / (max_val - min_val + 1e-6)
        #print("dual_att_log_logits:", dual_att_log_logits[:10])

        
        #dual_node_att = self.sampling(dual_att_log_logits, epoch, training)

        #dual_node_att = F.gumbel_softmax(dual_att_log_logits, tau = 0.1 ,dim = -1)[:,0]
        dual_node_att = self.gumbel_sigmoid(dual_att_log_logits, tau = 0.1)[:,0]
        dual_node_att = dual_node_att.unsqueeze(-1)
        #dual_node_att = self.sampling(dual_att_log_logits, epoch, training)

        y_uv = primal_data.edge_label.float().to(dual_node_att.device)
        f1_loss = self.f1_sparsity_loss(dual_node_att, y_uv)

        dual_node_att_other = self.sampling(dual_att_log_logits, epoch, training)

        if self.dual_learn_edge_att:
            if is_undirected(dual_data.edge_index):
                trans_idx, trans_val = transpose(dual_data.edge_index, dual_node_att, None, None, coalesced=False)
                trans_val_perm = reorder_like(trans_idx, dual_data.edge_index, trans_val)
                dual_edge_att = (dual_node_att + trans_val_perm) / 2
            else:
                dual_edge_att = dual_node_att
        else:
            dual_edge_att = self.lift_node_att_to_edge_att(dual_node_att, dual_data.edge_index)

        if self.primal_learn_edge_att:
            if is_undirected(primal_data.edge_index):
                trans_idx, trans_val = transpose(primal_data.edge_index, primal_node_att, None, None, coalesced=False)
                trans_val_perm = reorder_like(trans_idx, primal_data.edge_index, trans_val)
                primal_edge_att = (primal_node_att + trans_val_perm) / 2
            else:
                primal_edge_att = primal_node_att
        else:
            primal_edge_att = self.lift_node_att_to_edge_att(primal_node_att, primal_data.edge_index)
            old_primal_edge_att = self.lift_node_att_to_edge_att(primal_node_att, primal_data.edge_index)

        if (epoch > 50):
            primal_edge_att = 0.3 * dual_node_att + (1 - 0.3) * primal_edge_att

        primal_clf_logits = self.primal_clf(primal_data.x, primal_data.edge_index, primal_data.batch,
                                            edge_attr=primal_data.edge_attr, edge_atten=primal_edge_att)


        dual_clf_logits = self.dual_clf(dual_data.x, dual_data.edge_index, dual_data.batch,
                                    edge_attr=dual_data.edge_attr, edge_atten=dual_edge_att)

        batch = primal_data.batch.cpu().numpy()
        edge_index = primal_data.edge_index.cpu().numpy()
        primal_att = primal_edge_att.detach().cpu().numpy()

        dual_att = dual_node_att.detach().cpu().numpy()

        num_graphs = batch.max() + 1

        alpha = 0.3

        comb_att = alpha * dual_node_att + (1 - alpha) * old_primal_edge_att

        ground_truth_mask = primal_data.edge_label.cpu().numpy()

        loss, loss_dict = self.__loss__(primal_edge_att, dual_edge_att, primal_clf_logits, dual_clf_logits, primal_data.y, dual_data.y, dual_att_log_logits, epoch)

        #loss += self.lamb * (att_strength_penalty + att_var_penalty)
        #print("loss: ", loss)
        #print("f1_loss: ", f1_loss)
        loss += f1_loss
        #print("final loss: ", loss)
        #input('loss')


        # for g in range(15):
        #     # 1. Get node indices in graph g
        #     node_mask = (batch == g)
        #     node_indices = np.where(node_mask)[0]
        #     node_id_map = {global_idx: local_idx for local_idx, global_idx in enumerate(node_indices)}
        #     num_nodes = len(node_indices)

        #     # 2. Mask edges that are fully inside graph g
        #     edge_mask = np.isin(edge_index[0], node_indices) & np.isin(edge_index[1], node_indices)
        #     graph_edges = edge_index[:, edge_mask]
        #     primal_graph_att = primal_att[edge_mask]
        #     dual_graph_att = dual_att[edge_mask]
        #     comb_graph_att = comb_att[edge_mask]
        #     gt_graph_mask = ground_truth_mask[edge_mask]  

        #     # 3. Initialize empty attention matrix
        #     primal_att_matrix = np.full((num_nodes, num_nodes), np.nan)
        #     dual_att_matrix = np.full((num_nodes, num_nodes), np.nan)
        #     comb_att_matrix = np.full((num_nodes, num_nodes), np.nan)
        #     gt_matrix = np.full((num_nodes, num_nodes), np.nan)

        #     # 4. Fill matrix (use local node indices)
        #     for idx in range(graph_edges.shape[1]):
        #         src_global = graph_edges[0, idx]
        #         dst_global = graph_edges[1, idx]
        #         src_local = node_id_map[src_global]
        #         dst_local = node_id_map[dst_global]
        #         primal_att_matrix[src_local, dst_local] = primal_graph_att[idx]
        #         primal_att_matrix[dst_local, src_local] = primal_graph_att[idx]

        #         dual_att_matrix[src_local, dst_local] = dual_graph_att[idx]
        #         dual_att_matrix[dst_local, src_local] = dual_graph_att[idx]

        #         comb_att_matrix[src_local, dst_local] = comb_graph_att[idx]
        #         comb_att_matrix[dst_local, src_local] = comb_graph_att[idx]

        #         # if gt_graph_mask[idx] == 1:  # or True depending on dtype
        #         #     gt_matrix[src_local, dst_local] = 1
        #         #     gt_matrix[dst_local, src_local] = 1  # if undirected

        #     # gt_edge_indices = (gt_graph_mask == 1).nonzero()[0]
        #     # src_nodes = edge_index[0, gt_edge_indices]
        #     # tgt_nodes = edge_index[1, gt_edge_indices]

        #     # 5. Plot heatmap
        #     fig, ax = plt.subplots(figsize=(6, 5))
        #     sns.heatmap(primal_att_matrix, cmap='coolwarm', center=0, square=True)

        #     # ax.scatter(tgt_nodes + 0.5, src_nodes + 0.5, facecolors='none', edgecolors='green', s=100, linewidth=2, label='GT edges')

        #     plt.title(f"Primal Edge Attention Heatmap - Graph {g}")
        #     fig.canvas.manager.set_window_title(f"Primal Edge Att Graph {g}")
        #     plt.xlabel("Target Primal Node")
        #     plt.ylabel("Source Primal Node")
        #     plt.tight_layout()
        #     plt.show()
        #     plt.close()

        #     fig, ax = plt.subplots(figsize=(6, 5))
        #     sns.heatmap(dual_att_matrix, cmap='coolwarm', center=0, square=True)

        #     # ax.scatter(tgt_nodes + 0.5, src_nodes + 0.5, facecolors='none', edgecolors='green', s=100, linewidth=2, label='GT edges')

        #     plt.title(f"Dual Node Attention Heatmap - Graph {g}")
        #     fig.canvas.manager.set_window_title(f"Dual Node Att Graph {g}")
        #     plt.xlabel("Target Primal Node (local)")
        #     plt.ylabel("Source Primal Node (local)")
        #     plt.tight_layout()
        #     plt.show()
        #     plt.close()

        #     fig, ax = plt.subplots(figsize=(6, 5))
        #     sns.heatmap(comb_att_matrix, cmap='coolwarm', center=0, square=True)

        #     # gt_contour = np.nan_to_num(gt_matrix)
        #     # ax.contour(gt_contour, colors='green', linewidths=1)

        #     # ax.scatter(tgt_nodes + 0.5, src_nodes + 0.5, facecolors='none', edgecolors='green', s=100, linewidth=2, label='GT edges')
            
        #     plt.title(f"Comb Attention Heatmap - Graph {g}")
        #     fig.canvas.manager.set_window_title(f"Comb Att Graph {g}")
        #     plt.xlabel("Target Primal Node (local)")
        #     plt.ylabel("Source Primal Node (local)")
        #     plt.tight_layout()
        #     plt.show()
        #     plt.close()
            ##input(f"Continue after viewing Graph {g}?")

        # fig = plt.figure(figsize=(6,5))
        # sns.heatmap(primal_edge_att.detach().cpu().numpy(), cmap='coolwarm', center=0)
        # plt.title('Primal Edge Attention Heatmap')
        # fig.canvas.manager.set_window_title(f"Primal Edge Att Epoch {epoch}")
        # plt.show()
        # plt.close()

        # fig = plt.figure(figsize=(6,5))
        # sns.heatmap(dual_node_att.detach().cpu().numpy(), cmap='coolwarm', center=0)
        # plt.title('Dual Node Attention Heatmap')
        # fig.canvas.manager.set_window_title(f"Dual Node Att Epoch {epoch}")
        # plt.show()
        # plt.close()

        # fig = plt.figure(figsize=(6,5))
        # sns.heatmap(comb_att.detach().cpu().numpy(), cmap='coolwarm', center=0)
        # plt.title('Comb Attention Heatmap')
        # fig.canvas.manager.set_window_title(f"Comb Att Epoch {epoch}")
        # plt.show()
        # plt.close()
        if (epoch % 10 == 0):
            ground_truth_mask = primal_data.edge_label

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Get binary mask as indices
            gt_indices = torch.nonzero(ground_truth_mask == 1)  # returns tensor of shape [N, 1]
            gt_indices = gt_indices.view(-1).cpu().numpy()     


            # Primal Edge Attention
            sns.heatmap(primal_edge_att.detach().cpu().numpy().reshape(1, -1), ax=axes[0, 0], cmap='coolwarm', center=0)
            axes[0, 0].scatter(gt_indices, [0]*len(gt_indices), marker='x', color='green', label='Ground Truth')
            axes[0, 0].set_title('Primal Edge Attention')

            # Dual Node Attention
            sns.heatmap(dual_node_att.detach().cpu().numpy().reshape(1, -1), ax=axes[0, 1], cmap='coolwarm', center=0)
            axes[0, 1].scatter(gt_indices, [0]*len(gt_indices), marker='x', color='green', label='Ground Truth')
            axes[0, 1].set_title('Dual Node Attention')

            sns.heatmap(dual_node_att_other.detach().cpu().numpy().reshape(1, -1), ax=axes[1,0], cmap='coolwarm', center=0)
            axes[1, 0].scatter(gt_indices, [0]*len(gt_indices), marker='x', color='green', label='Ground Truth')
            axes[1, 0].set_title('Dual Node Attention w/o gumbel')

            # Combined Attention
            sns.heatmap(comb_att.detach().cpu().numpy().reshape(1, -1), ax=axes[1, 1], cmap='coolwarm', center=0)
            axes[1, 1].scatter(gt_indices, [0]*len(gt_indices), marker='x', color='green', label='Ground Truth')
            axes[1, 1].set_title('Combined Attention')

            # Final layout
            plt.suptitle(f'Attention Visualizations with Ground Truth - Epoch {epoch}')
            plt.tight_layout()
            plt.show()

        return primal_edge_att, loss, loss_dict, primal_clf_logits
        
        #Attempt 3.0

        # if epoch <= 50:
        #     # Primal 
        #     primal_emb = self.primal_clf.get_emb(primal_data.x, primal_data.edge_index, batch=primal_data.batch, edge_attr=primal_data.edge_attr)
        #     primal_att_log_logits = self.primal_extractor(primal_emb, primal_data.edge_index, primal_data.batch)
        #     primal_node_att = self.sampling(primal_att_log_logits, epoch, training)

        #     if self.primal_learn_edge_att:
        #         if is_undirected(primal_data.edge_index):
        #             trans_idx, trans_val = transpose(primal_data.edge_index, primal_node_att, None, None, coalesced=False)
        #             trans_val_perm = reorder_like(trans_idx, primal_data.edge_index, trans_val)
        #             primal_edge_att = (primal_node_att + trans_val_perm) / 2
        #         else:
        #             primal_edge_att = primal_node_att
        #     else:
        #         primal_edge_att = self.lift_node_att_to_edge_att(primal_node_att, primal_data.edge_index)

        #     primal_clf_logits = self.primal_clf(primal_data.x, primal_data.edge_index, primal_data.batch,
        #                                         edge_attr=primal_data.edge_attr, edge_atten=primal_edge_att)
        #     primal_loss, primal_loss_dict = self.__loss__(primal_edge_att, primal_clf_logits, primal_data.y, epoch)

        #     # Dual 
        #     dual_emb = self.dual_clf.get_emb(dual_data.x, dual_data.edge_index, batch=dual_data.batch, edge_attr=dual_data.edge_attr)
        #     dual_att_log_logits = self.dual_extractor(dual_emb, dual_data.edge_index, dual_data.batch)
        #     dual_node_att = self.sampling(dual_att_log_logits, epoch, training)

        #     if self.dual_learn_edge_att:
        #         if is_undirected(dual_data.edge_index):
        #             trans_idx, trans_val = transpose(dual_data.edge_index, dual_node_att, None, None, coalesced=False)
        #             trans_val_perm = reorder_like(trans_idx, dual_data.edge_index, trans_val)
        #             dual_edge_att = (dual_node_att + trans_val_perm) / 2
        #         else:
        #             dual_edge_att = dual_node_att
        #     else:
        #         dual_edge_att = self.lift_node_att_to_edge_att(dual_node_att, dual_data.edge_index)

        #     dual_clf_logits = self.dual_clf(dual_data.x, dual_data.edge_index, dual_data.batch,
        #                                 edge_attr=dual_data.edge_attr, edge_atten=dual_edge_att)
        #     dual_loss, dual_loss_dict = self.__loss__(dual_edge_att, dual_clf_logits, dual_data.y, epoch)

        #     input(f"primal_loss: {primal_loss}, dual_loss: {dual_loss}")

        #     lambda_att = 0.5  # tune this hyperparameter
        #     att_loss = F.mse_loss(dual_node_att, primal_edge_att.detach())
        #     loss = primal_loss + dual_loss + lambda_att * att_loss

        #     loss_dict = primal_loss_dict.copy()  
        #     loss_dict.update(dual_loss_dict)

        #     return primal_edge_att, loss, loss_dict, primal_clf_logits
        # else:
        #     primal_emb = self.primal_clf.get_emb(primal_data.x, primal_data.edge_index, batch=primal_data.batch, edge_attr=primal_data.edge_attr)
        #     primal_att_log_logits = self.primal_extractor(primal_emb, primal_data.edge_index, primal_data.batch)
        #     primal_node_att = self.sampling(primal_att_log_logits, epoch, training)

        #     primal_edge_att_from_nodes = self.lift_node_att_to_edge_att(primal_node_att, primal_data.edge_index)

        #     dual_emb = self.dual_clf.get_emb(dual_data.x, dual_data.edge_index, batch=dual_data.batch, edge_attr=dual_data.edge_attr)
        #     dual_att_log_logits = self.dual_extractor(dual_emb, dual_data.edge_index, dual_data.batch)
        #     dual_node_att = self.sampling(dual_att_log_logits, epoch, training)

        #     dual_edge_att = self.lift_node_att_to_edge_att(dual_node_att, dual_data.edge_index)

        #     alpha = 0.5  # tune this hyperparam
        #     primal_edge_att = alpha * dual_node_att + (1 - alpha) * primal_edge_att_from_nodes

        #     primal_clf_logits = self.primal_clf(primal_data.x, primal_data.edge_index, primal_data.batch,
        #                                         edge_attr=primal_data.edge_attr, edge_atten=primal_edge_att)
        #     primal_loss, primal_loss_dict = self.__loss__(primal_edge_att, primal_clf_logits, primal_data.y, epoch)

        #     dual_clf_logits = self.dual_clf(dual_data.x, dual_data.edge_index, dual_data.batch,
        #                                 edge_attr=dual_data.edge_attr, edge_atten=dual_edge_att)
        #     dual_loss, dual_loss_dict = self.__loss__(dual_edge_att, dual_clf_logits, dual_data.y, epoch)

        #     lambda_att = 0.5  # tune this hyperparameter
        #     att_loss = F.mse_loss(dual_node_att, primal_edge_att.detach())
        #     loss = primal_loss + dual_loss + lambda_att * att_loss

        # batch = primal_data.batch.cpu().numpy()
        # edge_index = primal_data.edge_index.cpu().numpy()
        # primal_att = primal_edge_att.detach().cpu().numpy()

        # dual_att = dual_node_att.detach().cpu().numpy()

        # num_graphs = batch.max() + 1

        # alpha = 0.5
        # comb_att = alpha * dual_node_att + (1 - alpha) * primal_edge_att

        # for g in range(15):
        #     # 1. Get node indices in graph g
        #     node_mask = (batch == g)
        #     node_indices = np.where(node_mask)[0]
        #     node_id_map = {global_idx: local_idx for local_idx, global_idx in enumerate(node_indices)}
        #     num_nodes = len(node_indices)

        #     # 2. Mask edges that are fully inside graph g
        #     edge_mask = np.isin(edge_index[0], node_indices) & np.isin(edge_index[1], node_indices)
        #     graph_edges = edge_index[:, edge_mask]
        #     primal_graph_att = primal_att[edge_mask]
        #     dual_graph_att = dual_att[edge_mask]
        #     comb_graph_att = comb_att[edge_mask]

        #     # 3. Initialize empty attention matrix
        #     primal_att_matrix = np.full((num_nodes, num_nodes), np.nan)
        #     dual_att_matrix = np.full((num_nodes, num_nodes), np.nan)
        #     comb_att_matrix = np.full((num_nodes, num_nodes), np.nan)

        #     # 4. Fill matrix (use local node indices)
        #     for idx in range(graph_edges.shape[1]):
        #         src_global = graph_edges[0, idx]
        #         dst_global = graph_edges[1, idx]
        #         src_local = node_id_map[src_global]
        #         dst_local = node_id_map[dst_global]
        #         primal_att_matrix[src_local, dst_local] = primal_graph_att[idx]
        #         primal_att_matrix[dst_local, src_local] = primal_graph_att[idx]

        #         dual_att_matrix[src_local, dst_local] = dual_graph_att[idx]
        #         dual_att_matrix[dst_local, src_local] = dual_graph_att[idx]

        #         comb_att_matrix[src_local, dst_local] = comb_graph_att[idx]
        #         comb_att_matrix[dst_local, src_local] = comb_graph_att[idx]

        #     # 5. Plot heatmap
        #     fig = plt.figure(figsize=(6, 5))
        #     sns.heatmap(primal_att_matrix, cmap='coolwarm', center=0, square=True)
        #     plt.title(f"Primal Edge Attention Heatmap - Graph {g}")
        #     fig.canvas.manager.set_window_title(f"Primal Edge Att Graph {g}")
        #     plt.xlabel("Target Primal Node")
        #     plt.ylabel("Source Primal Node")
        #     plt.tight_layout()
        #     plt.show()
        #     plt.close()

        #     fig = plt.figure(figsize=(6, 5))
        #     sns.heatmap(dual_att_matrix, cmap='coolwarm', center=0, square=True)
        #     plt.title(f"Dual Node Attention Heatmap - Graph {g}")
        #     fig.canvas.manager.set_window_title(f"Dual Node Att Graph {g}")
        #     plt.xlabel("Target Primal Node (local)")
        #     plt.ylabel("Source Primal Node (local)")
        #     plt.tight_layout()
        #     plt.show()
        #     plt.close()

        #     fig = plt.figure(figsize=(6, 5))
        #     sns.heatmap(comb_att_matrix, cmap='coolwarm', center=0, square=True)
        #     plt.title(f"Comb Attention Heatmap - Graph {g}")
        #     fig.canvas.manager.set_window_title(f"Comb Att Graph {g}")
        #     plt.xlabel("Target Primal Node (local)")
        #     plt.ylabel("Source Primal Node (local)")
        #     plt.tight_layout()
        #     plt.show()
        #     plt.close()
        #     #input(f"Continue after viewing Graph {g}?")

        # fig = plt.figure(figsize=(6,5))
        # sns.heatmap(primal_edge_att.detach().cpu().numpy(), cmap='coolwarm', center=0)
        # plt.title('Primal Edge Attention Heatmap')
        # fig.canvas.manager.set_window_title(f"Primal Edge Att Epoch {epoch}")
        # plt.show()
        # plt.close()

        # fig = plt.figure(figsize=(6,5))
        # sns.heatmap(dual_node_att.detach().cpu().numpy(), cmap='coolwarm', center=0)
        # plt.title('Dual Node Attention Heatmap')
        # fig.canvas.manager.set_window_title(f"Dual Node Att Epoch {epoch}")
        # plt.show()
        # plt.close()

        # fig = plt.figure(figsize=(6,5))
        # sns.heatmap(comb_att.detach().cpu().numpy(), cmap='coolwarm', center=0)
        # plt.title('Comb Attention Heatmap')
        # fig.canvas.manager.set_window_title(f"Comb Att Epoch {epoch}")
        # plt.show()
        # plt.close()

        #     loss_dict = primal_loss_dict.copy()  
        #     loss_dict.update(dual_loss_dict)

        #     return primal_edge_att, loss, loss_dict, primal_clf_logits
                
    @torch.no_grad()
    def dual_eval_one_batch(self, primal_data, dual_data, epoch):
        self.primal_extractor.eval()
        self.primal_clf.eval()

        self.dual_extractor.eval()
        self.dual_clf.eval()

        att, loss, loss_dict, clf_logits = self.dual_forward_pass(primal_data, dual_data, epoch, training=False)
        return att.data.cpu().reshape(-1), loss_dict, clf_logits.data.cpu()

    def dual_train_one_batch(self, primal_data, dual_data, epoch): #implement dual primal in forward pass or in GNN Layer
        self.primal_extractor.train()
        self.primal_clf.train()

        self.dual_extractor.train()
        self.dual_clf.train()

        # att, loss, loss_dict, clf_logits = self.forward_pass(data, epoch, training=True)
        att, loss, loss_dict, clf_logits = self.dual_forward_pass(primal_data, dual_data, epoch, training=True)
        self.primal_optimizer.zero_grad()
        self.dual_optimizer.zero_grad()
        loss.backward()
        self.primal_optimizer.step()
        self.dual_optimizer.step()
        return att.data.cpu().reshape(-1), loss_dict, clf_logits.data.cpu()

    def dual_run_one_epoch(self, primal_data_loader, dual_data_loader, epoch, phase, use_edge_attr):
        primal_loader_len = len(primal_data_loader)
        dual_loader_len = len(dual_data_loader)

        run_one_batch = self.dual_train_one_batch if phase == 'train' else self.dual_eval_one_batch
        phase = 'test ' if phase == 'test' else phase  # align tqdm desc bar

        all_loss_dict = {}
        all_exp_labels, all_att, all_clf_labels, all_clf_logits, all_precision_at_k, all_delta_kl = ([] for i in range(6))

        pbar = tqdm(zip(primal_data_loader, dual_data_loader), total=primal_loader_len)

        for idx, (primal_data, dual_data) in enumerate(pbar):
            primal_data = process_data(primal_data, use_edge_attr)
            dual_data = process_data(dual_data, use_edge_attr)
            att, loss_dict, clf_logits = run_one_batch(primal_data.to(self.primal_device), dual_data.to(self.dual_device), epoch)

            exp_labels = primal_data.edge_label.data.cpu()

            delta_kl = self.get_delta_kl(exp_labels, att)
            all_loss_dict['delta_kl'] = all_loss_dict.get('delta_kl', 0) + delta_kl

            precision_at_k = self.get_precision_at_k(att, exp_labels, self.primal_k, primal_data.batch, primal_data.edge_index)
            desc, _, _, _, _, _ = self.log_epoch(epoch, phase, loss_dict, exp_labels, att, precision_at_k,
                                                 primal_data.y.data.cpu(), clf_logits, delta_kl, batch=True)
            for k, v in loss_dict.items():
                all_loss_dict[k] = all_loss_dict.get(k, 0) + v

            all_delta_kl.append(delta_kl)
            all_exp_labels.append(exp_labels), all_att.append(att), all_precision_at_k.extend(precision_at_k)
            all_clf_labels.append(primal_data.y.data.cpu()), all_clf_logits.append(clf_logits)

            if idx == primal_loader_len - 1:
                all_exp_labels, all_att = torch.cat(all_exp_labels), torch.cat(all_att),
                all_clf_labels, all_clf_logits = torch.cat(all_clf_labels), torch.cat(all_clf_logits)

                for k, v in all_loss_dict.items():
                    all_loss_dict[k] = v / primal_loader_len
                desc, att_auroc, precision, clf_acc, clf_roc, avg_loss = self.log_epoch(epoch, phase, all_loss_dict, all_exp_labels, all_att,
                                                                                        all_precision_at_k, all_clf_labels, all_clf_logits, delta_kl=all_delta_kl, batch=False)
            pbar.set_description(desc)
        return att_auroc, precision, clf_acc, clf_roc, avg_loss

    def train(self, primal_loaders, dual_loaders, primal_test_set, dual_test_set, metric_dict, use_edge_attr): #just keeping same because didn't change much
        viz_set = self.get_viz_idx(primal_test_set, self.primal_dataset_name)

        viz_set = self.get_viz_idx(dual_test_set, self.dual_dataset_name)

        primal_graph_embeds = []
        dual_graph_embeds = []

        for epoch in range(self.primal_epochs):
            train_res = self.dual_run_one_epoch(primal_loaders['train'], dual_loaders['train'], epoch, 'train', use_edge_attr)
            valid_res = self.dual_run_one_epoch(primal_loaders['valid'], dual_loaders['valid'], epoch, 'valid', use_edge_attr)
            test_res = self.dual_run_one_epoch(primal_loaders['test'], dual_loaders['test'], epoch, 'test', use_edge_attr)
            self.primal_writer.add_scalar('gsat_train/lr', get_lr(self.primal_optimizer), epoch) #need dual?

            assert len(train_res) == 5
            main_metric_idx = 3 if 'ogb' in self.primal_dataset_name else 2  # clf_roc or clf_acc
            if self.primal_scheduler is not None:
                self.primal_scheduler.step(valid_res[main_metric_idx])

            r = self.primal_fix_r if self.primal_fix_r else self.get_r(self.primal_decay_interval, self.primal_decay_r, epoch, final_r=self.primal_final_r, init_r=self.primal_init_r)
            if (r == self.primal_final_r or self.primal_fix_r) and epoch > 10 and ((valid_res[main_metric_idx] > metric_dict['metric/best_clf_valid'])
                                                                     or (valid_res[main_metric_idx] == metric_dict['metric/best_clf_valid']
                                                                         and valid_res[4] < metric_dict['metric/best_clf_valid_loss'])):

                metric_dict = {'metric/best_clf_epoch': epoch, 'metric/best_clf_valid_loss': valid_res[4],
                               'metric/best_clf_train': train_res[main_metric_idx], 'metric/best_clf_valid': valid_res[main_metric_idx], 'metric/best_clf_test': test_res[main_metric_idx],
                               'metric/best_x_roc_train': train_res[0], 'metric/best_x_roc_valid': valid_res[0], 'metric/best_x_roc_test': test_res[0],
                               'metric/best_x_precision_train': train_res[1], 'metric/best_x_precision_valid': valid_res[1], 'metric/best_x_precision_test': test_res[1]}
                save_checkpoint(self.primal_clf, self.primal_model_dir, model_name='gsat_clf_epoch_' + str(epoch))
                save_checkpoint(self.primal_extractor, self.primal_model_dir, model_name='gsat_att_epoch_' + str(epoch))

            for metric, value in metric_dict.items():
                metric = metric.split('/')[-1]
                self.primal_writer.add_scalar(f'gsat_best/{metric}', value, epoch)

            if self.primal_num_viz_samples != 0 and (epoch % self.primal_viz_interval == 0 or epoch == epoch - 1):
                if self.primal_multi_label:
                    raise NotImplementedError
                for idx, tag in viz_set:
                    self.visualize_results(primal_test_set, dual_test_set, idx, epoch, tag, use_edge_attr)

            if epoch == self.primal_epochs - 1:
                save_checkpoint(self.primal_clf, self.primal_model_dir, model_name='gsat_clf_epoch_' + str(epoch))
                save_checkpoint(self.primal_extractor, self.primal_model_dir, model_name='gsat_att_epoch_' + str(epoch))

            print(f'[Seed {self.primal_random_state}, Epoch: {epoch}]: Best Epoch: {metric_dict["metric/best_clf_epoch"]}, '
                  f'Best Val Pred ACC/ROC: {metric_dict["metric/best_clf_valid"]:.3f}, Best Test Pred ACC/ROC: {metric_dict["metric/best_clf_test"]:.3f}, '
                  f'Best Test X AUROC: {metric_dict["metric/best_x_roc_test"]:.3f}')
            print('====================================')
            print('====================================')

        return metric_dict

    def log_epoch(self, epoch, phase, loss_dict, exp_labels, att, precision_at_k, clf_labels, clf_logits, delta_kl, batch):
        desc = f'[Seed {self.primal_random_state}, Epoch: {epoch}]: gsat_{phase}........., ' if batch else f'[Seed {self.primal_random_state}, Epoch: {epoch}]: gsat_{phase} finished, '
        for k, v in loss_dict.items():
            if not batch:
                self.primal_writer.add_scalar(f'gsat_{phase}/{k}', v, epoch)
            desc += f'{k}: {v:.3f}, '

        eval_desc, att_auroc, precision, clf_acc, clf_roc = self.get_eval_score(epoch, phase, exp_labels, att, precision_at_k, clf_labels, clf_logits, delta_kl, batch)
        desc += eval_desc
        return desc, att_auroc, precision, clf_acc, clf_roc, loss_dict['pred']

    def get_eval_score(self, epoch, phase, exp_labels, att, precision_at_k, clf_labels, clf_logits, delta_kl, batch):
        clf_preds = get_preds(clf_logits, self.primal_multi_label)
        clf_acc = 0 if self.primal_multi_label else (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]

        if batch:
            return f'clf_acc: {clf_acc:.3f}', None, None, None, None

        precision_at_k = np.mean(precision_at_k)
        delta_kl = np.mean(delta_kl)

        clf_roc = 0
        if 'ogb' in self.primal_dataset_name:
            evaluator = Evaluator(name='-'.join(self.primal_dataset_name.split('_')))
            clf_roc = evaluator.eval({'y_pred': clf_logits, 'y_true': clf_labels})['rocauc']

        att_auroc, bkg_att_weights, signal_att_weights = 0, att, att
        if np.unique(exp_labels).shape[0] > 1:
            att_auroc = roc_auc_score(exp_labels, att)
            bkg_att_weights = att[exp_labels == 0]
            signal_att_weights = att[exp_labels == 1]

        self.primal_writer.add_histogram(f'gsat_{phase}/bkg_att_weights', bkg_att_weights, epoch)
        self.primal_writer.add_histogram(f'gsat_{phase}/signal_att_weights', signal_att_weights, epoch)
        self.primal_writer.add_scalar(f'gsat_{phase}/clf_acc/', clf_acc, epoch)
        self.primal_writer.add_scalar(f'gsat_{phase}/clf_roc/', clf_roc, epoch)
        self.primal_writer.add_scalar(f'gsat_{phase}/att_auroc/', att_auroc, epoch)
        self.primal_writer.add_scalar(f'gsat_{phase}/precision@{self.primal_k}/', precision_at_k, epoch)
        self.primal_writer.add_scalar(f'gsat_{phase}/avg_bkg_att_weights/', bkg_att_weights.mean(), epoch)
        self.primal_writer.add_scalar(f'gsat_{phase}/avg_signal_att_weights/', signal_att_weights.mean(), epoch)
        self.primal_writer.add_scalar(f'gsat_{phase}/delta_kl', delta_kl, epoch)
        self.primal_writer.add_pr_curve(f'PR_Curve/gsat_{phase}/', exp_labels, att, epoch)

        desc = f'clf_acc: {clf_acc:.3f}, clf_roc: {clf_roc:.3f}, ' + \
       f'att_roc: {att_auroc:.3f}, att_prec@{self.primal_k}: {precision_at_k:.3f}, ' + \
       f'delta_kl: {delta_kl:.3f}'
        return desc, att_auroc, precision_at_k, clf_acc, clf_roc

    def get_precision_at_k(self, att, exp_labels, k, batch, edge_index):
        precision_at_k = []
        for i in range(batch.max()+1):
            nodes_for_graph_i = batch == i
            edges_for_graph_i = nodes_for_graph_i[edge_index[0]] & nodes_for_graph_i[edge_index[1]]
            labels_for_graph_i = exp_labels[edges_for_graph_i]
            mask_log_logits_for_graph_i = att[edges_for_graph_i]
            precision_at_k.append(labels_for_graph_i[np.argsort(-mask_log_logits_for_graph_i)[:k]].sum().item() / k)
        return precision_at_k
    
    def get_delta_kl(self, exp_labels, att, eps=1e-6):
        p = exp_labels.float().clamp(min=eps, max=1-eps)
        r_uv = att.clamp(min=eps, max=1-eps)
        r = r_uv.mean().clamp(min=eps, max=1-eps)

        delta_kl = p * torch.log(r_uv / r) + (1 - p) * torch.log((1 - r_uv) / (1 - r))

        return delta_kl.sum().item()


    def get_viz_idx(self, test_set, dataset_name):
        y_dist = test_set.data.y.numpy().reshape(-1)
        num_nodes = np.array([each.x.shape[0] for each in test_set])
        classes = np.unique(y_dist)
        res = []
        for each_class in classes:
            tag = 'class_' + str(each_class)
            if dataset_name == 'Graph-SST2':
                condi = (y_dist == each_class) * (num_nodes > 5) * (num_nodes < 10)  # in case too short or too long
                candidate_set = np.nonzero(condi)[0]
            else:
                candidate_set = np.nonzero(y_dist == each_class)[0]
            idx = np.random.choice(candidate_set, self.primal_num_viz_samples, replace=False)
            res.append((idx, tag))
        return res

    def visualize_results(self, primal_test_set, dual_test_set, idx, epoch, tag, use_edge_attr):
        primal_viz_set = primal_test_set[idx]
        dual_viz_set = dual_test_set[idx]

        primal_data = next(iter(DataLoader(primal_viz_set, batch_size=len(idx), shuffle=False)))
        primal_data = process_data(primal_data, use_edge_attr)

        dual_data = next(iter(DataLoader(dual_viz_set, batch_size=len(idx), shuffle=False)))
        dual_data = process_data(dual_data, use_edge_attr)

        print("AGHHHH WHY DO YOU DO THIS TO MEEEE")
        batch_att, _, clf_logits = self.dual_eval_one_batch(primal_data.to(self.primal_device), dual_data.to(self.dual_device), epoch)
        imgs = []
        for i in tqdm(range(len(primal_viz_set))):
            mol_type, coor = None, None
            if self.primal_dataset_name == 'mutag':
                node_dict = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S', 8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
                mol_type = {k: node_dict[v.item()] for k, v in enumerate(primal_viz_set[i].node_type)}
            elif self.primal_dataset_name == 'Graph-SST2':
                mol_type = {k: v for k, v in enumerate(primal_viz_set[i].sentence_tokens)}
                num_nodes = primal_data.x.shape[0]
                x = np.linspace(0, 1, num_nodes)
                y = np.ones_like(x)
                coor = np.stack([x, y], axis=1)
            elif self.primal_dataset_name == 'ogbg_molhiv':
                element_idxs = {k: int(v+1) for k, v in enumerate(primal_viz_set[i].x[:, 0])}
                mol_type = {k: Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), int(v)) for k, v in element_idxs.items()}
            elif self.primal_dataset_name == 'mnist':
                raise NotImplementedError

            node_subset = primal_data.batch == i
            _, edge_att = subgraph(node_subset, primal_data.edge_index, edge_attr=batch_att)

            node_label = primal_viz_set[i].node_label.reshape(-1) if primal_viz_set[i].get('node_label', None) is not None else torch.zeros(primal_viz_set[i].x.shape[0])
            fig, img = visualize_a_graph(primal_viz_set[i].edge_index, edge_att, node_label, self.primal_dataset_name, norm=self.primal_viz_norm_att, mol_type=mol_type, coor=coor)
            #plt.show()
            #plt.close(fig)
            imgs.append(img)
        imgs = np.stack(imgs)
        self.primal_writer.add_images(tag, imgs, epoch, dataformats='NHWC')

    def get_r(self, decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r

    def sampling(self, att_log_logits, epoch, training):
        att = self.concrete_sample(att_log_logits, temp=1, training=training)
        return att

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att

    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern


class ExtractorMLP(nn.Module):

    def __init__(self, hidden_size, shared_config, type):
        super().__init__()
        if(type == "primal"):
            self.primal_learn_edge_att = shared_config['learn_edge_att']
            dropout_p = shared_config['extractor_dropout_p']

            if self.primal_learn_edge_att:
                self.primal_feature_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=dropout_p)
            else:
                self.primal_feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=dropout_p)
        elif (type == "dual"):
            self.dual_learn_edge_att = shared_config['learn_edge_att']
            dropout_p = shared_config['extractor_dropout_p']

            if self.dual_learn_edge_att:
                self.dual_feature_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=dropout_p)
            else:
                self.dual_feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=dropout_p)

    def forward(self, emb, edge_index, batch, type):
        if (type == "primal"):
            if self.primal_learn_edge_att:
                col, row = edge_index
                f1, f2 = emb[col], emb[row]
                f12 = torch.cat([f1, f2], dim=-1)
                att_log_logits = self.primal_feature_extractor(f12, batch[col])
            else:
                att_log_logits = self.primal_feature_extractor(emb, batch)
            return att_log_logits
        elif (type == "dual"):
            if self.dual_learn_edge_att:
                col, row = edge_index
                f1, f2 = emb[col], emb[row]
                f12 = torch.cat([f1, f2], dim=-1)
                att_log_logits = self.dual_feature_extractor(f12, batch[col])
            else:
                att_log_logits = self.dual_feature_extractor(emb, batch)
            return att_log_logits


def train_gsat_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, method_name, device, random_state):
    print('====================================')
    print('====================================')
    print(f'[INFO] Using device: {device}')
    print(f'[INFO] Using random_state: {random_state}')
    print(f'[INFO] Using dataset: {dataset_name}')
    print(f'[INFO] Using model: {model_name}')

    set_seed(random_state)

    #DUAL
    dual_log_dir = log_dir / 'DUAL'
    dual_data_dir = data_dir / '_dual'
    dual_dataset_name = dataset_name + "_dual"
    #DUAL

    model_config = local_config['model_config']
    data_config = local_config['data_config']
    method_config = local_config[f'{method_name}_config']
    shared_config = local_config['shared_config']
    assert model_config['model_name'] == model_name
    assert method_config['method_name'] == method_name

    #DUAL
    dual_model_config = local_config['model_config']
    dual_data_config = local_config['data_config']
    dual_method_config = local_config[f'{method_name}_config']
    dual_shared_config = local_config['shared_config']
    assert dual_model_config['model_name'] == model_name
    assert dual_method_config['method_name'] == method_name
    #DUAL

    batch_size, splits = data_config['batch_size'], data_config.get('splits', None)
    loaders, primal_test_set, x_dim, edge_attr_dim, num_class, aux_info = get_data_loaders(data_dir, dataset_name, batch_size, splits, random_state, data_config.get('mutag_x', False))

    #DUAL
    dual_loaders, dual_test_set, dual_x_dim, dual_edge_attr_dim, dual_num_class, dual_aux_info = get_data_loaders(data_dir, dual_dataset_name, batch_size, splits, random_state, data_config.get('mutag_x', False))
    
    #input(dual_x_dim)
    #DUAL

    model_config['deg'] = aux_info['deg'] # degree histogram
    primal_model = get_model(x_dim, edge_attr_dim, num_class, aux_info['multi_label'], model_config, device)

    #DUAL
    dual_model_config['deg'] = dual_aux_info['deg']
    dual_model = get_model(dual_x_dim, dual_edge_attr_dim, dual_num_class, dual_aux_info['multi_label'], dual_model_config, device)
    #input(dual_x_dim)
    #DUAL

    print('====================================')
    print('====================================')

    log_dir.mkdir(parents=True, exist_ok=True)
    if not method_config['from_scratch']:
        print('[INFO] Pretraining the model...')
        train_clf_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, device, random_state,
                           model=primal_model, loaders=loaders, num_class=num_class, aux_info=aux_info)
        pretrain_epochs = local_config['model_config']['pretrain_epochs'] - 1
        load_checkpoint(primal_model, model_dir=log_dir, model_name=f'epoch_{pretrain_epochs}')
    else:
        print('[INFO] Training both the model and the attention from scratch...')

    #DUAL
    dual_log_dir.mkdir(parents=True, exist_ok=True)
    if not dual_method_config['from_scratch']:
        print('[DUAL INFO] Pretraining the model...')
        train_clf_one_seed(local_config, dual_data_dir, dual_log_dir, model_name, dual_dataset_name, device, random_state,
                           model=dual_model, loaders=dual_loaders, num_class=dual_num_class, aux_info=dual_aux_info)
        pretrain_epochs = local_config['model_config']['pretrain_epochs'] - 1
        load_checkpoint(dual_model, model_dir=dual_log_dir, model_name=f'epoch_{pretrain_epochs}_dual')
    else:
        print('[DUAL INFO] Training both the model and the attention from scratch...')
    #DUAL

    extractor = ExtractorMLP(model_config['hidden_size'], shared_config, "primal").to(device) #this is the parameter to be changed 
    lr, wd = method_config['lr'], method_config.get('weight_decay', 0)
    primal_optimizer = torch.optim.Adam(list(extractor.parameters()) + list(primal_model.parameters()), lr=lr, weight_decay=wd)

    scheduler_config = method_config.get('scheduler', {})
    primal_scheduler = None if scheduler_config == {} else ReduceLROnPlateau(primal_optimizer, mode='max', **scheduler_config)

    writer = Writer(log_dir=log_dir)
    hparam_dict = {**model_config, **data_config}
    hparam_dict = {k: str(v) if isinstance(v, (dict, list)) else v for k, v in hparam_dict.items()}
    metric_dict = deepcopy(init_metric_dict)
    writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)

    #DUAL
    dual_extractor = ExtractorMLP(dual_model_config['hidden_size'], dual_shared_config, "dual").to(device)
    dual_lr, dual_wd = dual_method_config['lr'], dual_method_config.get('weight_decay', 0)
    dual_optimizer = torch.optim.Adam(list(dual_extractor.parameters()) + list(dual_model.parameters()), lr=dual_lr, weight_decay=dual_wd)

    dual_scheduler_config = dual_method_config.get('scheduler', {})
    dual_scheduler = None if dual_scheduler_config == {} else ReduceLROnPlateau(dual_optimizer, mode='max', **dual_scheduler_config)

    dual_writer = Writer(log_dir=dual_log_dir)
    dual_hparam_dict = {**dual_model_config, **dual_data_config}
    dual_hparam_dict = {k: str(v) if isinstance(v, (dict, list)) else v for k, v in dual_hparam_dict.items()}
    dual_metric_dict = deepcopy(init_metric_dict)
    dual_writer.add_hparams(hparam_dict=dual_hparam_dict, metric_dict=dual_metric_dict)

    gsat = GSAT(primal_model, extractor, primal_optimizer, primal_scheduler, writer, device, log_dir, dataset_name, num_class, aux_info['multi_label'], random_state, method_config, shared_config, model_config, dual_model, dual_extractor, dual_optimizer, dual_scheduler, dual_writer, device, dual_log_dir, dual_dataset_name, dual_num_class, dual_aux_info['multi_label'], random_state, dual_method_config, dual_shared_config, dual_model_config)
    metric_dict = gsat.train(loaders, dual_loaders, primal_test_set, dual_test_set, metric_dict, model_config.get('use_edge_attr', True))
    writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)
    return hparam_dict, metric_dict


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train GSAT')
    parser.add_argument('--dataset', type=str, help='dataset used')
    parser.add_argument('--backbone', type=str, help='backbone model used')
    parser.add_argument('--cuda', type=int, help='cuda device id, -1 for cpu')
    args = parser.parse_args()
    dataset_name = args.dataset
    model_name = args.backbone
    cuda_id = args.cuda

    torch.set_num_threads(5)
    config_dir = Path('./configs')
    method_name = 'GSAT'

    print('====================================')
    print('====================================')
    print(f'[INFO] Running {method_name} on {dataset_name} with {model_name}')
    print('====================================')

    global_config = yaml.safe_load((config_dir / 'global_config.yml').open('r'))
    local_config_name = get_local_config_name(model_name, dataset_name)

    #input(config_dir / local_config_name)

    local_config = yaml.safe_load((config_dir / local_config_name).open('r'))

    data_dir = Path(global_config['data_dir'])
    num_seeds = global_config['num_seeds']

    time = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
    device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu')

    metric_dicts = []
    for random_state in range(num_seeds):
        log_dir = data_dir / dataset_name / 'logs' / (time + '-' + dataset_name + '-' + model_name + '-seed' + str(random_state) + '-' + method_name)
        hparam_dict, metric_dict = train_gsat_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, method_name, device, random_state)
        metric_dicts.append(metric_dict)

    # alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    # lambdas = [0.01, 0.1, 0.3, 0.5]

    # best_score = -float("inf")
    # best_params = None

    # for alpha in alphas:
    #     for lamb in lambdas:
    #         print(f"Trying alpha={alpha}, lambda_att={lamb}")
    #         metric_dicts = []

    #         # Run training across seeds
    #         for random_state in range(num_seeds):

    #             log_dir = data_dir / dataset_name / 'logs' / (
    #                 time + '-' + dataset_name + '-' + model_name + 
    #                 f'-alpha{alpha}-lambda{lamb}-seed{random_state}-' + method_name
    #             )

    #             hparam_dict, metric_dict = train_gsat_one_seed(
    #                 local_config, data_dir, log_dir, model_name, dataset_name, 
    #                 method_name, device, random_state, alpha, lamb
    #             )
    #             metric_dicts.append(metric_dict)

    #         avg_metric = sum([m['metric/best_clf_valid'] for m in metric_dicts]) / len(metric_dicts)

    #         print(f"alpha={alpha}, lambda_att={lamb}, val acc = {avg_metric:.4f}")

    #         if avg_metric > best_score:
    #             best_score = avg_metric
    #             best_params = (alpha, lamb)

    # print(f"Best params: alpha={best_params[0]}, lambda_att={best_params[1]}")
    # print(f"Best validation accuracy: {best_score:.4f}")


    log_dir = data_dir / dataset_name / 'logs' / (time + '-' + dataset_name + '-' + model_name + '-seed99-' + method_name + '-stat')
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = Writer(log_dir=log_dir)
    write_stat_from_metric_dicts(hparam_dict, metric_dicts, writer)


if __name__ == '__main__':
    main()
