import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from DensityTree import RandomDensityTree
from scipy import stats
from scipy.stats import multivariate_normal

# def eval_splits(x, dim, node_ids):
#     x_node_dim = x[node_ids,dim]
#     sorted_ids = [y for(x,y) in sorted(zip(x_node_dim, node_ids), key=lambda x: x[0])]
#     best_i_gain = 0
#     split_idx = 0
#     for i in range(len(sorted_ids)):
#         i_gain = information_gain(x,sorted_ids[0:i], sorted_ids[i:])
#         if(i_gain>best_i_gain):
#             best_i_gain = i_gain
#             split_idx = i
#     # left < split_val, right >= split_val
#     split_val = x[sorted_ids[split_idx],dim]
#     return split_val, sorted_ids[0:split_idx], sorted_ids[split_idx:]

# def information_gain(x,left_ids, right_ids):
#     log_det_node_cov = np.log(np.det(np.cov(x[list(left_ids) + list(right_ids)])))
#     ratio_left = len(left_ids)/(len(left_ids) + len(right_ids))
#     log_det_left_cov = np.log(np.det(np.cov(x[list(left_ids)])))
#     ratio_right = len(right_ids)/(len(left_ids) + len(right_ids))
#     log_det_right_cov = np.log(np.det(np.cov(x[list(right_ids)])))
#     i_gain = log_det_node_cov - ratio_left*log_det_left_cov - ratio_right*log_det_right_cov
#     return i_gain

def generate_monte_carlo_sample(X, num_samples=1000000):
    """max_depth=10,num_splits=10,min_infogain=1.5
    Generate more sample points
    """
    samples = np.random.rand(num_samples,len(x[0]))
    d_mins = np.min(X,axis=0)
    d_maxs = np.max(X,axis=0)
    samples = np.add(np.multiply(samples,d_maxs-d_mins),d_mins)
    b_size = np.prod(d_maxs-d_mins)
    return samples, b_size

def partition_function(tree, X, t_dict):
    # generate a lot of samples in the bounds of the data and the size of the bounded shape
    samples, b_size = generate_monte_carlo_sample(x)
    # add gaussian probability dimension for those samples
    g_probs_samples = np.random.random(len(samples))
    # predict the target leaf nodes for all samples
    leaf_node_ids = np.array(tree.predict(samples))
    unique_leaf_ids = np.unique(leaf_node_ids)
    # compute the distribution integral over each leaf node
    g_ints = []
    for ln_id in unique_leaf_ids:
        leaf_node = tree.tree[ln_id]
        # mean_vec = leaf_node.mean
        # cov_mat = leaf_node.cov
        portion = t_dict[ln_id][0]

        mean_vec = t_dict[ln_id][1]
        cov_mat = t_dict[ln_id][2]
        g_probs = multivariate_normal.pdf(samples[leaf_node_ids==ln_id], mean_vec, cov_mat, allow_singular=True)
        g_cnt = np.sum(g_probs_samples[leaf_node_ids==ln_id]<=g_probs)
        g_ints.append(portion*(g_cnt/len(samples)*b_size))

    return np.sum(g_ints)

def get_bootstrap_indices(num_samples,max_length):
    samples = np.random.rand(num_samples)*max_length
    return samples.astype(int)



class DensityForest:

    def __init__(self, n_estimators, max_depth=10, num_splits=10, min_infogain=1.5, boostrap=True):
        self.forest = [RandomDensityTree(max_depth,num_splits,min_infogain) for i in range(n_estimators)]
        self.boostrap = boostrap

    def fit(self, X):
        if(not(type(X)=="numpy.ndarray")):
            X = np.array(X)

        self.p_funcs = []
        self.tree_dicts = []
        for tree in self.forest:
            tree_sample = X
            if(self.boostrap):
                tree_sample = X[get_bootstrap_indices(len(X),len(X))]
            tree.fit(tree_sample)

            # Traverse the tree and set the attributes for each node
            self.set_node_attributes(tree.root, X, len(X))

            t_dict = {}
            clusters = np.array(tree.predict(X))
            unique_cl = np.unique(clusters)
            for val in unique_cl:
                data = X[clusters[:]==val,:]
                # t_dict[val] = [len(data)/len(X), np.mean(data,axis=0), np.cov(data.T)]
                t_dict[val] = [len(data)/len(X), tree.tree[val].mean, tree.tree[val].cov]
            self.tree_dicts.append(t_dict)

            

            self.p_funcs.append(partition_function(tree,X,t_dict))
            print(self.p_funcs[-1])

    def set_node_attributes(self, node, X, N):
      # Set the bounding box
      node.bbox_min = np.min(X, axis=0)
      node.bbox_max = np.max(X, axis=0)

      if node.isLeaf:
          # Set the probability for leaf nodes
          node.p = len(X) / N
      else:
          # Recursively set the attributes for the child nodes
          X_left = X[X[:, node.feature] <= node.threshold]
          X_right = X[X[:, node.feature] > node.threshold]

          self.set_node_attributes(node.left, X_left, N)
          self.set_node_attributes(node.right, X_right, N)

          # Set the probabilities for non-leaf nodes
          node.p_left = node.left.p
          node.p = node.p_left + node.right.p

    
    def predict(self, X):
        leaf_preds = np.zeros((len(X), len(self.forest)))
        for i,tree in enumerate(self.forest):
            preds = np.array(tree.predict(X)).astype(float)

            unique_preds = np.unique(preds)
            for j,val in enumerate(unique_preds):
                leaf_idc = preds==val
                # p_vals = multivariate_normal.pdf(X[preds==val], mean=tree.tree[val].mean, cov=tree.tree[val].cov)
                norm_fac = self.tree_dicts[i][val][0]/self.p_funcs[i]
                # norm_fac = 1/self.p_funcs[i]

                p_vals = multivariate_normal.pdf(X[leaf_idc], mean=self.tree_dicts[i][val][1], cov=self.tree_dicts[i][val][2], allow_singular=True)


                preds[leaf_idc] = p_vals*norm_fac
            leaf_preds[:,i] = preds

        mean_p = np.mean(leaf_preds, axis=1)
        return mean_p
    
    def sample(self, n_samples):
        """
        generate samples from the forest
        """
        samples=[]
        # Draw a tree index uniformly
        tree_index = np.random.choice(len(self.forest), size = n_samples)
        for i in range(n_samples):
          tree = self.forest[tree_index[i]]

          # Start at the root node
          node = tree.root

          # Traverse the tree
          while not node.isLeaf:
              z = np.random.uniform(0, 1)

              # Traverse to the left child if z < p_left
              if z < node.p_left:
                  node = node.left
              else:
                  # Otherwise, traverse to the right child
                  z -= node.p_left
                  node = node.right

          # Sample x uniformly from the leaf's bounding box
          samples.append(np.random.uniform(node.bbox_min, node.bbox_max))
        return np.array(samples)




# create tree
df = RandomDensityTree(max_depth=1,num_splits=10,min_infogain=1.5)

# create dataset
mn = multivariate_normal(mean=[2,2], cov=[0.3,0.7])
dist1 = mn.rvs(100)
tar1 = multivariate_normal.pdf(dist1, np.mean(dist1,axis=0), np.cov(dist1.T))
mn = multivariate_normal(mean=[8,8], cov=[1,0.5])
dist2 = mn.rvs(100)
tar2 = multivariate_normal.pdf(dist2, np.mean(dist2,axis=0), np.cov(dist2.T))
x = np.concatenate((dist1,dist2),axis=0)
targets = np.concatenate((tar1,tar2),axis=0)

# shuffle the data
shuffle_idc = np.arange(len(x))
np.random.shuffle(shuffle_idc)
x = x[shuffle_idc,:]
targets = targets[shuffle_idc]

df.fit(x)

clusters = np.array(df.predict(x))
unique_cl = np.unique(clusters)
data = []
colors = cm.rainbow(np.linspace(0, 1, len(unique_cl)))
groups = []

for val in unique_cl:
    data.append(x[clusters==val,:])
    mean = np.mean(data[-1],axis=0)
    # print(mean)
    # print(df.tree[val].mean)

    cov = np.cov(data[-1].T)
    preds = multivariate_normal.pdf(data[-1], mean=mean, cov=cov)
    tars = targets[clusters==val]
    # for p,t in zip(preds,tars):
    #     print(p,t)

    groups.append(str(val))


d_forest = DensityForest(n_estimators=5,max_depth=10,num_splits=10,min_infogain=1.5,boostrap=False)
d_forest.fit(x)
preds = d_forest.predict(x)
# for p,t in zip(preds[:10],targets[:10]):
#     print(p,t)


x_axis = np.linspace(0,12,200)
y_axis = np.linspace(0,12,200)
X,Y = np.meshgrid(x_axis,y_axis)
grid_data = np.concatenate((X.reshape(40000,1),Y.reshape(40000,1)),axis=1)

clusters = np.array(df.predict(grid_data))
unique_cl = np.unique(clusters)
p_vals = np.zeros(len(grid_data))
for val in unique_cl:
    leaf_data = grid_data[clusters==val,:]
    # mean = np.mean(leaf_data,axis=0)
    # print(mean)
    mean = df.tree[val].mean
    print(mean)
    # cov = np.cov(leaf_data.T)
    # print(cov)
    cov = df.tree[val].cov
    print(cov)
    preds = multivariate_normal.pdf(leaf_data, mean=mean, cov=cov)
    p_vals[clusters==val] = preds

p_vals = d_forest.predict(grid_data)
plt.pcolormesh(X,Y,p_vals.reshape(np.shape(X)))
plt.show()

# Create plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for data, color, group in zip(data, colors, groups):
    ax.scatter(data[:,0], data[:,1], alpha=1.0, c=color, edgecolors='none', s=5, label=group)

plt.title('Matplot scatter plot')
plt.legend(loc=2)
plt.show()
