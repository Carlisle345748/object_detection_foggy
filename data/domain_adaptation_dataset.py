import torch.utils.data as data


class DAAspectRatioGroupedDataset(data.IterableDataset):
    """
    Batch data that have similar aspect ratio together.
    In this implementation, images whose aspect ratio < (or >) 1 will
    be batched together.
    This improves training speed because the images then need less padding
    to form a batch.

    It assumes the underlying dataset produces dicts with "width" and "height" keys.
    It will then produce a list of original dicts with length = batch_size,
    all with similar aspect ratios.
    """

    def __init__(self, source_dataloader, target_dataloader, source_batch_size, target_batch_size):
        """
        Args:
            source_dataloader: an iterable dataloader of source domain. Each element must be a dict with keys
                "width" and "height", which will be used to batch data.
            target_dataloader: an iterable dataloader of target domain. Each element must be a dict with keys
                "width" and "height", which will be used to batch data.
            source_batch_size(int): source domain batch size
            target_batch_size(int): target domain batch size
        """
        self.source = source_dataloader
        self.target = target_dataloader
        self.source_batch_size = source_batch_size
        self.target_batch_size = target_batch_size

        self.source_bucket = [[] for _ in range(2)]
        self.target_bucket = [[] for _ in range(2)]

    def __iter__(self):
        for s, t in zip(self.source, self.target):
            # source dataset
            sw, sh = s["width"], s["height"]
            s_bucket_id = 0 if sw > sh else 1
            s_bucket = self.source_bucket[s_bucket_id]
            s_bucket.append(s)

            # target dataset
            t.pop("instances")  # Remove annotations for target domain
            tw, th = t["width"], t["height"]
            t_bucket_id = 0 if tw > th else 1
            t_bucket = self.target_bucket[t_bucket_id]
            t_bucket.append(t)

            if len(s_bucket) == self.source_batch_size and \
                    len(self.target_bucket[s_bucket_id]) == self.target_batch_size:
                batch = (s_bucket[:], self.target_bucket[s_bucket_id][:])
                del s_bucket[:]
                del self.target_bucket[s_bucket_id][:]
                yield batch

            if len(t_bucket) == self.target_batch_size and \
                    len(self.source_bucket[t_bucket_id]) == self.source_batch_size:
                batch = (self.source_bucket[t_bucket_id][:], t_bucket[:])
                del t_bucket[:]
                del self.source_bucket[t_bucket_id][:]
                yield batch


class DAGroupedDataset(data.IterableDataset):
    def __init__(self, source_dataloader, target_dataloader):
        """
        Args:
            source_dataloader: an iterable dataloader of source domain. Each element must be a dict with keys
                "width" and "height", which will be used to batch data.
            target_dataloader: an iterable dataloader of target domain. Each element must be a dict with keys
                "width" and "height", which will be used to batch data.
        """
        self.source = source_dataloader
        self.target = target_dataloader

    def __iter__(self):
        for s, t in zip(self.source, self.target):
            yield s, t

