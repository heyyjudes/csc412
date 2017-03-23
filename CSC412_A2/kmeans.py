from test import *

plt.ion()

def distmat(p, q):
  """Computes pair-wise L2-distance between columns of p and q."""
  d, pn = p.shape
  d, qn = q.shape
  pmag = np.sum(p**2, axis=0).reshape(1, -1)
  qmag = np.sum(q**2, axis=0).reshape(1, -1)
  dist = qmag + pmag.T - 2 * np.dot(p.T, q)
  dist = (dist >= 0) * dist  # Avoid small negatives due to numerical errors.
  return np.sqrt(dist)

def KMeans(x, K, iters):
  """Cluster x into K clusters using K-Means.
  Inputs:
    x: Data matrix, with one data vector per column.
    K: Number of clusters.
    iters: Number of iterations of K-Means to run.
  Outputs:
    means: Cluster centers, with one cluster center in each column.
  """
  N = x.shape[1]
  perm = np.arange(N)
  np.random.shuffle(perm)
  means = x[:, perm[:K]]
  dist = np.zeros((K, N))
  for ii in xrange(iters):
    print('Kmeans iteration = %04d' % (ii+1))
    for k in xrange(K):
      dist[k, :] = distmat(means[:, k].reshape(-1, 1), x)
    assigned_class = np.argmin(dist, axis=0)
    for k in xrange(K):
      means[:, k] = np.mean(x[:, (assigned_class == k).nonzero()[0]], axis=1)

  return means

def ShowMeans(means, number=0):
  """Show the cluster centers as images."""
  plt.figure(number)
  plt.clf()
  for i in xrange(means.shape[1]):
    plt.subplot(1, means.shape[1], i+1)
    means_i = means[:, i].reshape(28, 28)
    plt.imshow(means_i)
  plt.draw()
  plt.savefig("figure1.png")
  raw_input('Press Enter.')



def main():
  K = 30
  iters = 100
  #inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('../toronto_face.npz')
  train_data = np.load("training_data.npy")
  train_labels = np.load("training_labels.npy")

  means = KMeans(np.transpose(train_data), K, iters)
  plot_images(np.transpose(means), plt)
  plt.show()
  plt.savefig("figure1.png")
  #ShowMeans(means, 0)

if __name__ == '__main__':
  main()
