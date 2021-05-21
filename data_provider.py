import torch
import numpy as np
from collections import defaultdict
from utils import softmax


class TrajDataloder:
    def __init__(self, data_dict, dist_index_dict, device, shuffle=True, gen_pairs=False, K=10, theta=1000):

        self.dist_index_dict = dist_index_dict
        self.history_trajs = np.array(data_dict.history_trajs)
        self.history_trajs_masks = np.array(data_dict.history_trajs_masks)
        self.masked_current_trajs = np.array(data_dict.masked_current_trajs)
        self.masked_indexes = np.array(data_dict.mask_indexes)
        self.masked_true_traj_points = np.array(data_dict.masked_true_traj_points)
        self.history_length = self.history_trajs.shape[1]
        self.length = len(self.history_trajs)
        self.shuffle = shuffle
        self.device = device
        self.gen_pairs = gen_pairs
        if self.gen_pairs:
            self.K = K
            # weight scaler
            self.theta = theta
            self.init_K_neighbors()

    def init_K_neighbors(self):
        print("init K neighbors")
        dist_dict_of_list = defaultdict(list)
        for gid1, gid2 in self.dist_index_dict:
            dist_dict_of_list[gid1].append((gid2, self.dist_index_dict[gid1,gid2]))
        self.topk_dict = dict()
        self.topk_dist_dict = dict()
        self.non_topk_dict = dict()
        for gid in dist_dict_of_list:
            sorted_list = list(sorted(dist_dict_of_list[gid], key=lambda x: x[1]))
            gid_list = list(map(lambda x: x[0], sorted_list))
            dist_list = list(map(lambda x: x[1], sorted_list))
            self.topk_dist_dict[gid] = softmax(np.array(dist_list[:self.K])/self.theta)
            self.topk_dict[gid] = gid_list[:self.K]
            self.non_topk_dict[gid] = gid_list[self.K:]

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.history_trajs = self.history_trajs[shuffled_arg]
            self.history_trajs_masks = self.history_trajs_masks[shuffled_arg]
            self.masked_current_trajs = self.masked_current_trajs[shuffled_arg]
            self.masked_indexes = self.masked_indexes[shuffled_arg]
            self.masked_true_traj_points = self.masked_true_traj_points[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def construct_graph(self, inputs):
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        for u_input in inputs:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))
            for i in range(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        return alias_inputs, A, items

    def make_dist_pairs(self, anchor_points):
        # For now, we fix the number of negative sample at one.
        anchor_points_no_pad = []
        pos_points = []
        pos_weights = []
        neg_points = []
        for point in anchor_points:
            try: # maybe the points in anchor_points are the indexes of pad, <m> and <*>, which should be ignored.
                pos_index = np.random.randint(len(self.topk_dict[point]))
                neg_index = np.random.randint(len(self.non_topk_dict[point]))
                pos_points.append(self.topk_dict[point][pos_index])
                pos_weights.append(self.topk_dist_dict[point][pos_index])
                neg_points.append(self.non_topk_dict[point][neg_index])
                anchor_points_no_pad.append(point)
            except:
                continue
        return anchor_points_no_pad, pos_points, neg_points, pos_weights

    def get_slice(self, i):
        # history_trajs: batch_size, history_length, seq_length
        # masked_current_trajs: batch_size, seq_length
        history_trajs, history_trajs_masks, masked_current_trajs, masked_indexes, targets = \
            self.history_trajs[i], self.history_trajs_masks[i], self.masked_current_trajs[i], self.masked_indexes[i], self.masked_true_traj_points[i]
        if self.gen_pairs:
            anchor_points = np.concatenate([np.unique(history_trajs), np.unique(masked_current_trajs)], 0)
            anchor_points, pos_points, neg_points, pos_weights = self.make_dist_pairs(anchor_points)
            anchor_points = torch.tensor(anchor_points, dtype=torch.long)
            pos_points = torch.tensor(pos_points, dtype=torch.long)
            neg_points = torch.tensor(neg_points, dtype=torch.long)
            pos_weights = torch.tensor(pos_weights, dtype=torch.float)
        # reshape history trajs: batch_size * history_length, seq_length
        reshaped_history_trajs = history_trajs.reshape(-1, history_trajs.shape[-1])
        reshaped_history_alias_inputs, reshaped_history_A, reshaped_history_items = self.construct_graph(reshaped_history_trajs)
        masked_current_alias_inputs, masked_current_A, masked_current_items = self.construct_graph(masked_current_trajs)

        # convert to tensor
        history_trajs_masks = torch.tensor(history_trajs_masks, dtype=torch.long)
        reshaped_history_alias_inputs = torch.tensor(reshaped_history_alias_inputs, dtype=torch.long)
        reshaped_history_A = torch.tensor(reshaped_history_A, dtype=torch.float)
        reshaped_history_items = torch.tensor(reshaped_history_items, dtype=torch.long)
        masked_current_alias_inputs = torch.tensor(masked_current_alias_inputs, dtype=torch.long)
        masked_current_A = torch.tensor(masked_current_A, dtype=torch.float)
        masked_current_items = torch.tensor(masked_current_items, dtype=torch.long)
        # batch_size, mask_num
        masked_indexes = torch.tensor(masked_indexes, dtype=torch.long)
        # batch_size * mask_num
        targets = torch.tensor(targets, dtype=torch.long).view(-1)

        if self.gen_pairs:
            batch = [history_trajs_masks, reshaped_history_alias_inputs, reshaped_history_A, reshaped_history_items, masked_current_alias_inputs,
                     masked_current_A, masked_current_items, masked_indexes, targets,
                     anchor_points, pos_points, neg_points, pos_weights]
        else:
            batch = [history_trajs_masks, reshaped_history_alias_inputs, reshaped_history_A, reshaped_history_items, masked_current_alias_inputs,
                     masked_current_A, masked_current_items, masked_indexes, targets]
        return self.to_device(batch)

    def to_device(self, data):
        return [d.to(self.device) for d in data]

