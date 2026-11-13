"""Exact L2 nearest-neighbour search on the GPU with torch, standing in for
faiss's GPU flat index when the installed faiss has no GPU support (no faiss
GPU wheel for the Python of the Kaggle image).

LoopSplat uses faiss in three places (src/utils/mapper_utils.py,
src/gsr/overlap.py, src/utils/utils.batch_search_faiss): an IndexFlatL2(3)
moved to the GPU, then add(points), search(queries, k) -> (squared L2
distances, ids), reset(), all on torch CUDA tensors. A flat index is a brute
force search, so this gives the same neighbours and distances (up to float
rounding and the order of exact ties); LoopSplat only thresholds the
distances. install() patches the faiss module in place; with a GPU faiss it
does nothing.
"""
import torch

QUERY_CHUNK = 1024  # queries per distance block: 1024 x (map points) floats


class TorchFlatL2:
    def __init__(self, d):
        self.d = d
        self.xb = None

    @property
    def ntotal(self):
        return 0 if self.xb is None else int(self.xb.shape[0])

    def add(self, x):
        x = torch.as_tensor(x).float()
        if x.ndim != 2 or x.shape[1] != self.d:
            raise ValueError(f"add: expected (n, {self.d}), got {tuple(x.shape)}")
        if not x.is_cuda:
            x = x.cuda()
        self.xb = x.contiguous() if self.xb is None else torch.cat([self.xb, x.contiguous()])

    def reset(self):
        self.xb = None

    def search(self, q, k):
        q = torch.as_tensor(q).float()
        device = q.device
        q = q.cuda() if not q.is_cuda else q
        if self.xb is None or self.xb.shape[0] == 0:
            raise RuntimeError("search on an empty index")
        kk = min(k, self.xb.shape[0])
        xb_sq = (self.xb * self.xb).sum(1)
        dists, ids = [], []
        for c in torch.split(q, QUERY_CHUNK):
            d2 = (c * c).sum(1, keepdim=True) + xb_sq[None] - 2.0 * (c @ self.xb.T)
            d2.clamp_(min=0.0)
            dd, ii = torch.topk(d2, kk, dim=1, largest=False, sorted=True)
            dists.append(dd)
            ids.append(ii)
        dist, idx = torch.cat(dists), torch.cat(ids).long()
        if kk < k:  # faiss pads missing neighbours with inf / -1
            pad = k - kk
            dist = torch.cat([dist, torch.full((dist.shape[0], pad), float("inf"), device=dist.device)], 1)
            idx = torch.cat([idx, torch.full((idx.shape[0], pad), -1, device=idx.device, dtype=idx.dtype)], 1)
        return dist.to(device), idx.to(device)


class _NoGpuResources:
    pass


def has_gpu_faiss(faiss):
    return hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0


def install(faiss):
    """Returns "faiss-gpu" when faiss has GPU support (nothing changed), else
    patches faiss and returns "torch-shim"."""
    if has_gpu_faiss(faiss):
        return "faiss-gpu"
    faiss.StandardGpuResources = _NoGpuResources
    faiss.index_cpu_to_gpu = lambda res, device, index, *a, **k: TorchFlatL2(index.d)
    faiss.index_cpu_to_all_gpus = lambda index, *a, **k: TorchFlatL2(index.d)
    return "torch-shim"
