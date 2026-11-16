"""A DataLoader stand-in that keeps a tensor dataset on the GPU.

The trajectory datasets are precomputed tensors, so a standard DataLoader spends
each step stacking 256 small tensors on the CPU and copying the batch to the
GPU. For a ~0.8M-parameter model that overhead, not the model, sets the step
time. DeviceLoader moves the dataset to the device once and gathers each batch
there.

Sampling matches DataLoader(shuffle=..., drop_last=False): each pass over the
data is a fresh uniformly random permutation, without replacement, ending with
a partial batch. The permutation stream comes from its own generator seeded by
`seed`, so batches are not bit-identical to a DataLoader run with the same seed,
only identically distributed.
"""
import torch


class DeviceLoader:
    def __init__(self, dataset, batch_size, shuffle, device, seed=0):
        self.dataset = dataset
        self.device = torch.device(device)
        self.tensors = tuple(t.to(self.device) for t in
                             (dataset.contexts, dataset.targets, dataset.r_labels))
        self.n = len(dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            order = torch.randperm(self.n, generator=self.generator, device=self.device)
        else:
            order = torch.arange(self.n, device=self.device)
        for i in range(0, self.n, self.batch_size):
            idx = order[i:i + self.batch_size]
            yield tuple(t[idx] for t in self.tensors)
