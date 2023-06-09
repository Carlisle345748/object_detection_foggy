import logging
import operator

import torch.utils.data as torchdata
from detectron2.data import DatasetFromList, MapDataset, get_detection_dataset_dicts, \
    DatasetMapper
from detectron2.data.build import worker_init_reset_seed, trivial_batch_collator
from detectron2.data.common import ToIterableDataset
from detectron2.data.samplers import RandomSubsetTrainingSampler, RepeatFactorTrainingSampler, TrainingSampler
from detectron2.utils.comm import get_world_size

from data.dataset_mapper import TeacherStudentDepthDatasetMapper
from data.domain_adaptation_dataset import DAAspectRatioGroupedDataset, DAGroupedDataset


def build_domain_adaptation_train_loader(cfg):
    """
    Build a Domain Adaptation dataloader

    Args:
        cfg: config
    Returns:
        A dataloader that return a batch of data. Each batch contains source domain images, source
        domain object detection labels, source domain depth map, and target domain images
    """
    mapper = TeacherStudentDepthDatasetMapper(cfg)

    source = train_dataset_from_config(cfg=cfg, dataset_name=cfg.DATASETS.TRAIN_SOURCE, mapper=mapper)
    target = train_dataset_from_config(cfg=cfg, dataset_name=cfg.DATASETS.TRAIN_TARGET, mapper=mapper)

    return build_domain_adaptation_data_loader(
        source=source,
        target=target,
        source_batch_size=cfg.SOLVER.IMS_PER_BATCH_SOURCE,
        target_batch_size=cfg.SOLVER.IMS_PER_BATCH_TARGET,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS
    )


def build_domain_adaptation_data_loader(
        source,
        target,
        source_batch_size,
        target_batch_size,
        aspect_ratio_grouping,
        num_workers
):
    """
    Build dataloader for Domain Adaptation

    Args:
        source (dict): a dictionary containing "dataset" and "sampler" of source dataset
        target (dict): a dictionary containing "dataset" and "sampler" of target dataset
        source_batch_size (int): batch size of source dataset
        target_batch_size (int): batch size of target dataset
        aspect_ratio_grouping (bool): whether to group data by aspect ratio
        num_workers (int): number of worker for dataloader

    Returns:
        Dataloader for Domain Adaptation

    """
    world_size = get_world_size()
    assert (
            source_batch_size > 0 and source_batch_size % world_size == 0
    ), "Source batch size ({}) must be divisible by the number of gpus ({}).".format(
        source_batch_size, world_size
    )
    assert (
            target_batch_size > 0 and target_batch_size % world_size == 0
    ), "Target batch size ({}) must be divisible by the number of gpus ({}).".format(
        target_batch_size, world_size
    )

    source_batch_size = source_batch_size // world_size
    target_batch_size = target_batch_size // world_size

    # source dataset
    source_dataset = ToIterableDataset(source["dataset"], source["sampler"])
    source_dataloader = build_dataloader(
        dataset=source_dataset,
        batch_size=source_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers
    )

    # target dataset
    target_dataset = ToIterableDataset(target["dataset"], target["sampler"])
    target_dataloader = build_dataloader(
        dataset=target_dataset,
        batch_size=target_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers
    )

    if aspect_ratio_grouping:
        return DAAspectRatioGroupedDataset(
            source_dataloader=source_dataloader,
            target_dataloader=target_dataloader,
            source_batch_size=source_batch_size,
            target_batch_size=target_batch_size
        )
    else:
        return DAGroupedDataset(
            source_dataloader=source_dataloader,
            target_dataloader=target_dataloader
        )


def build_dataloader(dataset, batch_size, aspect_ratio_grouping, num_workers):
    """
    Builder a dataloader for a dataset

    Args:
        dataset: the dataset to load
        batch_size (int): batch size for the dataset. Only used when aspect_ratio_grouping=False
        aspect_ratio_grouping (bool): whether to group data by aspect ratio
        num_workers (int): number of workers for dataloader

    Returns:
        A dataloader for the given dataset
    """
    if aspect_ratio_grouping:
        data_loader = torchdata.DataLoader(
            dataset,
            num_workers=num_workers,
            collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        return data_loader
    else:
        return torchdata.DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=True,
            num_workers=num_workers,
            collate_fn=trivial_batch_collator,
            worker_init_fn=worker_init_reset_seed,
        )


def build_sampler(cfg, dataset):
    """
    Build a dataset sampler according to config

    Args:
        cfg: config
        dataset: dataset to sample

    Returns:
        The dataset sampler
    """
    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    if isinstance(dataset, torchdata.IterableDataset):
        logger.info("Not using any sampler since the dataset is IterableDataset.")
        sampler = None
    else:
        logger.info("Using training sampler {}".format(sampler_name))
        if sampler_name == "TrainingSampler":
            sampler = TrainingSampler(len(dataset))
        elif sampler_name == "RepeatFactorTrainingSampler":
            repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                dataset, cfg.DATALOADER.REPEAT_THRESHOLD
            )
            sampler = RepeatFactorTrainingSampler(repeat_factors)
        elif sampler_name == "RandomSubsetTrainingSampler":
            sampler = RandomSubsetTrainingSampler(
                len(dataset), cfg.DATALOADER.RANDOM_SUBSET_RATIO
            )
        else:
            raise ValueError("Unknown training sampler: {}".format(sampler_name))

    return sampler


def train_dataset_from_config(cfg, *, dataset_name, mapper=None, sampler=None):
    """
    Build a dataset from config

    Args:
        cfg: config
        dataset_name: the name of the dataset
        mapper: dataset mapper. if not given, build a mapper according to config
        sampler: dataset sampler. If not given, build a sampler according to config

    Returns:
        A training dataset
    """
    dataset = get_detection_dataset_dicts(
        dataset_name,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    )

    if mapper is None:
        mapper = DatasetMapper(cfg)

    if sampler is None:
        sampler = build_sampler(cfg, dataset)

    # Transform list to dataset
    dataset = DatasetFromList(dataset)  # Build dataset from list
    dataset = MapDataset(dataset, mapper)  # Apply dataset mapper

    return {"dataset": dataset, "sampler": sampler}
