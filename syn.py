import numpy as np
from utils import cal_edge_homophily


class SynGraphGenerator():
    def __init__(self, N, P, P_link, C, D) -> None:
        self.N = N
        self.P = P
        self.P_link = P_link
        self.C = C
        self.D = D

    def generate_graph(self, mode='homo'):
        H_homo, H_heter = self.cal_H(t=5)
        if mode == 'homo':
            edge_list, features, labels = self.generate_base_graph(H_homo, name='homo')
        elif mode == 'heter':
            edge_list, features, labels = self.generate_base_graph(H_heter, name='heter')
        else:
            edge_list1, features1, labels1 = self.generate_base_graph(H_homo, name='homo')
            edge_list2, features2, labels2 = self.generate_base_graph(H_heter, name='heter')
            edge_list, features, labels = self.combine(edge_list1, features1, labels1, edge_list2, features2, labels2)
        print('-' * 16, 'Graph: {}'.format(mode), '-' * 16)
        print('Edge shape: ', edge_list.shape)
        print('Feature shape: ', features.shape)
        print('Edge homophily: {:.4}'.format(cal_edge_homophily(edge_list, labels)))
        return edge_list, features, labels

    def generate_base_graph(self, H, name='default'):
        edge_list = []
        features = np.zeros((self.N, self.D))
        labels = np.zeros(self.N).astype(int)
        HP = H * self.P
        # generate labels randomly
        for u in range(self.N):
            label = np.random.randint(0, self.C)
            labels[u] = label

        # generate edges using H
        for u in range(self.N):
            u_class = labels[u]
            for v_class in range(self.C):
                nodes_v_class = np.nonzero(labels == v_class)[0]            # node index whose class equals v_class
                nodes_v_class_num = np.count_nonzero(labels == v_class)     # number of nodes whose class equals v_class
                # Randomly select nodes_v_class_num * HP[u_class][v_class] nodes to build edges.
                # In other words, using p[u_class][v_class] to build edge with class specified probability.
                edges = np.random.permutation(nodes_v_class)[: int(nodes_v_class_num * HP[u_class][v_class])]
                for v in edges:
                    edge_list.append([u, v])

        edge_list = np.array(edge_list).T

        for c in range(self.C):
            class_num = np.count_nonzero(labels == c)
            feature_c = np.random.normal(loc=c, scale=1, size=(class_num, self.D))
            features[labels == c] = feature_c
        
        return edge_list, features.astype(np.float32), labels

    # Use t to control H_homo's non-diagonal values to fall in [0. 1/(t+1)]
    def cal_H(self, t=5):
        H_random = np.random.rand(self.C, self.C)
        H_heter = np.ones((self.C, self.C)) - np.diag(H_random.diagonal())
        H_random = (np.random.rand(self.C, self.C) + t) / (t + 1)   # ([0, 1] + 5) / 6 falls in [5/6, 1]
        H_homo = np.ones((self.C, self.C)) - (H_random - np.diag(H_random.diagonal()))  # Non diagonal is in [0, 1/6]
        regular = H_heter.sum() / H_homo.sum()
        H_heter /= regular
        return H_homo, H_heter

    # combine two graph using random generate links with probability P_link
    def combine(self, edge_list1, features1, labels1, edge_list2, features2, labels2):
        # reset graph2's node index to begin from N
        edge_list2 = edge_list2 + self.N
        # random sample some nodes in both graph to link edge
        edge_list_between = []

        for u in range(self.N):
            nodes_v = np.random.permutation(self.N)[: int(self.N * self.P_link)] + self.N
            for v in nodes_v:
                edge_list_between.append([u, v])
                edge_list_between.append([v, u])

        edge_list_between = np.array(edge_list_between).T
        
        print('Link edges : ', edge_list_between.shape[1])

        edge_list = np.concatenate([edge_list1, edge_list2, edge_list_between], axis=1)
        features = np.concatenate([features1, features2], axis=0)
        labels = np.concatenate([labels1, labels2], axis=0)
        return edge_list, features, labels


if __name__ == '__main__':
    N = 100
    P = 0.5
    P_link = 0.05
    C = 5
    D = 50
    data_generator = SynGraphGenerator(N, P, P_link, C, D)
    edge_list, features, labels = data_generator.generate_graph(mode='combined')
    from visual.visualize import plot_tsne
    plot_tsne(features, labels, 'syn-f', seed=42, pca_dim=0, alpha=0.8)
