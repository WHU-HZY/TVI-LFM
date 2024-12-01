import numpy as np
from torch.utils.data.sampler import Sampler

def GenIdx(train_color_label, train_thermal_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k, v in enumerate(train_color_label) if v == unique_label_color[i]]
        color_pos.append(tmp_pos)

    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k, v in enumerate(train_thermal_label) if v == unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)

    return color_pos, thermal_pos

class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, num_pos, batchSize):
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)

        N = np.maximum(len(train_color_label), len(train_thermal_label))
        for j in range(int(N / (batchSize * num_pos)) + 1):
            batch_idx = np.random.choice(uni_label, batchSize, replace=False)
            for i in range(batchSize):
                sample_color = np.random.choice(color_pos[batch_idx[i]], num_pos)
                sample_thermal = np.random.choice(thermal_pos[batch_idx[i]], num_pos)

                if j == 0 and i == 0:
                    index1 = sample_color
                    index2 = sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))

        self.index1 = index1
        self.index2 = index2
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N

'''

class IdentitySampler(Sampler):

    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, batchSize, per_img):
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)

        sample_color = np.arange(batchSize)
        sample_thermal = np.arange(batchSize)
        N = np.maximum(len(train_color_label), len(train_thermal_label))

        # per_img = 4
        per_id = batchSize / per_img
        for j in range(N // batchSize + 1):
            batch_idx = np.random.choice(uni_label, int(per_id), replace=False)

            for s, i in enumerate(range(0, batchSize, per_img)):
                sample_color[i:i + per_img] = np.random.choice(color_pos[batch_idx[s]], per_img, replace=False)
                sample_thermal[i:i + per_img] = np.random.choice(thermal_pos[batch_idx[s]], per_img, replace=False)

            if j == 0:
                index1 = sample_color
                index2 = sample_thermal
            else:
                index1 = np.hstack((index1, sample_color))
                index2 = np.hstack((index2, sample_thermal))

        self.index1 = index1
        self.index2 = index2
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N
        '''