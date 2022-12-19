import torch
from pathlib import Path
from abc import abstractmethod
import numpy as np
import multiprocessing
from utils import augmentations, experiment_manager, geofiles


class AbstractMultimodalCDDataset(torch.utils.data.Dataset):

    def __init__(self, cfg: experiment_manager.CfgNode, run_type: str):
        super().__init__()
        self.cfg = cfg
        self.run_type = run_type
        self.root_path = Path(cfg.PATHS.DATASET)

        self.metadata = geofiles.load_json(self.root_path / f'metadata.json')

        self.s1_band_indices = cfg.DATALOADER.S1_BANDS
        self.s2_band_indices = cfg.DATALOADER.S2_BANDS

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def _load_s1_img(self, aoi_id: str, year: int, month: int) -> np.ndarray:
        file = self.root_path / aoi_id / 's1' / f's1_{aoi_id}_{year}_{month:02d}.tif'
        img, _, _ = geofiles.read_tif(file)
        img = np.clip(img[:, :, self.s1_band_indices], 0, 1)
        return np.nan_to_num(img).astype(np.float32)

    def _load_s2_img(self, aoi_id: str, year: int, month: int) -> np.ndarray:
        file = self.root_path / aoi_id / 's2' / f's2_{aoi_id}_{year}_{month:02d}.tif'
        img, _, _ = geofiles.read_tif(file)
        img = np.clip(img[:, :, self.s2_band_indices], 0, 1)
        return np.nan_to_num(img).astype(np.float32)

    def _load_building_label(self, aoi_id: str, year: int, month: int) -> np.ndarray:
        file = self.root_path / aoi_id / 'buildings' / f'buildings_{aoi_id}_{year}_{month:02d}.tif'
        label, _, _ = geofiles.read_tif(file)
        label = label > 0
        return np.nan_to_num(label).astype(np.float32)

    def _load_change_label(self, aoi_id: str, year_t1: int, month_t1: int, year_t2: int, month_t2) -> np.ndarray:
        building_t1 = self._load_building_label(aoi_id, year_t1, month_t1)
        building_t2 = self._load_building_label(aoi_id, year_t2, month_t2)
        change = np.logical_and(building_t1 == 0, building_t2 == 1)
        return change.astype(np.float32)

    def get_aoi_ids(self) -> list:
        return list(set([s['aoi_id'] for s in self.samples]))

    def get_geo(self, aoi_id: str) -> tuple:
        timestamps = self.metadata[aoi_id]
        timestamps = [(ts['year'], ts['month']) for ts in timestamps if ts['s1']]
        year, month = timestamps[0]
        file = self.root_path / aoi_id / 's1' / f's1_{aoi_id}_{year}_{month:02d}.tif'
        _, transform, crs = geofiles.read_tif(file)
        return transform, crs

    def load_s2_rgb(self, aoi_id: str, year: int, month: int) -> np.ndarray:
        file = self.root_path / aoi_id / 's2' / f's2_{aoi_id}_{year}_{month:02d}.tif'
        img, _, _ = geofiles.read_tif(file)
        img = np.clip(img[:, :, [2, 1, 0]] / 0.3, 0, 1)
        return np.nan_to_num(img).astype(np.float32)


# dataset for urban extraction with building footprints
class MultimodalCDDataset(AbstractMultimodalCDDataset):

    def __init__(self, cfg: experiment_manager.CfgNode, run_type: str, no_augmentations: bool = False,
                 dataset_mode: str = None, disable_multiplier: bool = False, disable_unlabeled: bool = False):
        super().__init__(cfg, run_type)

        self.dataset_mode = cfg.DATALOADER.DATASET_MODE if dataset_mode is None else dataset_mode
        self.include_building_labels = cfg.DATALOADER.INCLUDE_BUILDING_LABELS

        # handling transformations of data
        self.no_augmentations = no_augmentations
        self.transform = augmentations.compose_transformations(cfg, no_augmentations)

        # loading labeled samples (sn7 train set) and subset to run type aoi ids
        if run_type == 'training':
            self.aoi_ids = list(cfg.DATASET.TRAINING_IDS)
        elif run_type == 'validation':
            self.aoi_ids = list(cfg.DATASET.VALIDATION_IDS)
        else:
            self.aoi_ids = list(cfg.DATASET.TEST_IDS)

        self.labeled = [True] * len(self.aoi_ids)

        # TODO:  unlabeled data for semi-supervised learning
        if (cfg.DATALOADER.INCLUDE_UNLABELED or cfg.DATALOADER.INCLUDE_UNLABELED_VALIDATION) and not disable_unlabeled:
            aoi_ids_unlabelled = []
            if cfg.DATALOADER.INCLUDE_UNLABELED:
                aoi_ids_unlabelled += list(cfg.DATASET.UNLABELED_IDS)
            if cfg.DATALOADER.INCLUDE_UNLABELED_VALIDATION:
                aoi_ids_unlabelled += list(cfg.DATASET.VALIDATION_IDS)
            aoi_ids_unlabelled = sorted(aoi_ids_unlabelled)
            self.aoi_ids.extend(aoi_ids_unlabelled)
            self.labeled.extend([False] * len(aoi_ids_unlabelled))

        if not disable_multiplier:
            self.aoi_ids = self.aoi_ids * cfg.DATALOADER.TRAINING_MULTIPLIER
            self.labeled = self.labeled * cfg.DATALOADER.TRAINING_MULTIPLIER

        manager = multiprocessing.Manager()
        self.unlabeled_ids = manager.list(list(self.cfg.DATASET.UNLABELED_IDS))
        self.aoi_ids = manager.list(self.aoi_ids)
        self.labeled = manager.list(self.labeled)
        self.metadata = manager.dict(self.metadata)

        self.length = len(self.aoi_ids)

    def __getitem__(self, index):

        aoi_id = self.aoi_ids[index]
        labeled = self.labeled[index]
        timestamps = self.metadata[aoi_id]
        if labeled:
            timestamps = [(ts['year'], ts['month']) for ts in timestamps if ts['s1'] and ts['s2'] and ts['buildings'] and not ts['masked']]
        else:
            timestamps = [(ts['year'], ts['month']) for ts in timestamps if ts['s1'] and ts['s2']]

        if self.dataset_mode == 'first_last':
            indices = [0, -1]
        else:
            indices = sorted(np.random.randint(0, len(timestamps), size=2))

        # t1
        year_t1, month_t1 = timestamps[indices[0]]
        img_s1_t1 = self._load_s1_img(aoi_id, year_t1, month_t1)
        img_s2_t1 = self._load_s2_img(aoi_id, year_t1, month_t1)

        # t2
        year_t2, month_t2 = timestamps[indices[1]]
        img_s1_t2 = self._load_s1_img(aoi_id, year_t2, month_t2)
        img_s2_t2 = self._load_s2_img(aoi_id, year_t2, month_t2)

        if labeled:
            change = self._load_change_label(aoi_id, year_t1, month_t1, year_t2, month_t2)
            if self.include_building_labels:
                buildings_t1 = self._load_building_label(aoi_id, year_t1, month_t1)
                buildings_t2 = self._load_building_label(aoi_id, year_t2, month_t2)
                buildings = np.concatenate((buildings_t1, buildings_t2), axis=-1).astype(np.float32)
            else:
                buildings = np.zeros((change.shape[0], change.shape[1], 2), dtype=np.float32)
        else:
            change = np.zeros((img_s1_t1.shape[0], img_s1_t1.shape[1], 1), dtype=np.float32)
            buildings = np.zeros((change.shape[0], change.shape[1], 2), dtype=np.float32)

        # transformation, but this ain't pretty
        imgs = np.concatenate((img_s1_t1, img_s1_t2, img_s2_t1, img_s2_t2), axis=-1)
        imgs, buildings, change = self.transform((imgs, buildings, change))
        imgs_s1 = imgs[:2*len(self.s1_band_indices), ]
        img_s1_t1, img_s1_t2 = imgs_s1[:len(self.s1_band_indices), ], imgs_s1[len(self.s1_band_indices):, ]
        imgs_s2 = imgs[2*len(self.s1_band_indices):, ]
        img_s2_t1, img_s2_t2 = imgs_s2[:len(self.s2_band_indices), ], imgs_s2[len(self.s2_band_indices):, ]

        if self.cfg.DATALOADER.INPUT_MODE == 's1':
            x_t1, x_t2 = img_s1_t1, img_s1_t2
        elif self.cfg.DATALOADER.INPUT_MODE == 's2':
            x_t1, x_t2 = img_s2_t1, img_s2_t2
        else:
            x_t1 = torch.concat((img_s1_t1, img_s2_t1), dim=0)
            x_t2 = torch.concat((img_s1_t2, img_s2_t2), dim=0)

        item = {
            'x_t1': x_t1,
            'x_t2': x_t2,
            'y_change': change,
            'aoi_id': aoi_id,
            'year_t1': year_t1,
            'month_t1': month_t1,
            'year_t2': year_t2,
            'month_t2': month_t2,
            'is_labeled': labeled,
        }

        if self.include_building_labels:
            buildings_t1, buildings_t2 = buildings[0, ], buildings[1, ]
            item['y_sem_t1'] = buildings_t1.unsqueeze(0)
            item['y_sem_t2'] = buildings_t2.unsqueeze(0)

        return item

    def get_index(self, aoi_id: str) -> int:
        for index, candidate_aoi_id in enumerate(self.aoi_ids):
            if aoi_id == candidate_aoi_id:
                return index
        return None

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'


class SimpleInferenceDataset(torch.utils.data.Dataset):

    def __init__(self, cfg: experiment_manager.CfgNode, site: str, t1: str, t2: str, patch_size: int = 256):

        self.cfg = cfg
        self.root_path = Path(cfg.PATHS.DATASET)

        self.site = site
        self.t1 = t1
        self.t2 = t2

        self.s1_band_indices = cfg.DATALOADER.S1_BANDS
        self.s2_band_indices = cfg.DATALOADER.S2_BANDS

        self.img_s1_t1 = self._load_s1_img(t1)
        self.img_s1_t2 = self._load_s1_img(t2)
        self.img_s2_t1 = self._load_s2_img(t1)
        self.img_s2_t2 = self._load_s2_img(t2)

        self.transform = augmentations.compose_transformations(cfg, True)

        self.patch_size = patch_size
        m, n, _ = self.img_s1_t1.shape
        self.m = m // patch_size
        self.n = n // patch_size

        self.patches = [(i, j) for j in range(self.n) for i in range(self.m)]

        self.length = len(self.patches)

    def __getitem__(self, index):

        i, j = self.patches[index]
        i_min, i_max = i * self.patch_size, (i + 1) * self.patch_size
        j_min, j_max = j * self.patch_size, (j + 1) * self.patch_size

        # t1
        patch_s1_t1 = self.img_s1_t1[i_min:i_max, j_min:j_max, :]
        patch_s2_t1 = self.img_s2_t1[i_min:i_max, j_min:j_max, :]

        # t2
        patch_s1_t2 = self.img_s1_t2[i_min:i_max, j_min:j_max, :]
        patch_s2_t2 = self.img_s2_t2[i_min:i_max, j_min:j_max, :]


        change_dummy = np.zeros((self.patch_size, self.patch_size, 1), dtype=np.float32)
        buildings_dummy = np.zeros((self.patch_size, self.patch_size, 2), dtype=np.float32)

        # transformation, but this ain't pretty
        patches = np.concatenate((patch_s1_t1, patch_s1_t2, patch_s2_t1, patch_s2_t2), axis=-1)
        patches, _, _ = self.transform((patches, buildings_dummy, change_dummy))
        patches_s1 = patches[:2 * len(self.s1_band_indices), ]
        patch_s1_t1, patch_s1_t2 = patches_s1[:len(self.s1_band_indices), ], patches_s1[len(self.s1_band_indices):, ]
        patches_s2 = patches[2 * len(self.s1_band_indices):, ]
        patch_s2_t1, patch_s2_t2 = patches_s2[:len(self.s2_band_indices), ], patches_s2[len(self.s2_band_indices):, ]

        if self.cfg.DATALOADER.INPUT_MODE == 's1':
            x_t1, x_t2 = patch_s1_t1, patch_s1_t2
        elif self.cfg.DATALOADER.INPUT_MODE == 's2':
            x_t1, x_t2 = patch_s2_t1, patch_s2_t2
        else:
            x_t1 = torch.concat((patch_s1_t1, patch_s2_t1), dim=0)
            x_t2 = torch.concat((patch_s1_t2, patch_s2_t2), dim=0)

        item = {
            'x_t1': x_t1,
            'x_t2': x_t2,
            'i_min': i_min,
            'i_max': i_max,
            'j_min': j_min,
            'j_max': j_max,
        }

        return item

    def _load_s1_img(self, date: str) -> np.ndarray:
        file = self.root_path / f'sentinel1_{self.site}_{date}.tif'
        img, _, _ = geofiles.read_tif(file)
        img = np.clip(img[:, :, self.s1_band_indices], 0, 1)
        return np.nan_to_num(img).astype(np.float32)

    def _load_s2_img(self, date: str) -> np.ndarray:
        file = self.root_path / f'sentinel2_{self.site}_{date}.tif'
        img, _, _ = geofiles.read_tif(file)
        img = np.clip(img[:, :, self.s2_band_indices], 0, 1)
        return np.nan_to_num(img).astype(np.float32)

    def get_arr(self, dtype=np.uint8):
        height = self.m * self.patch_size
        width = self.n * self.patch_size
        return np.zeros((height, width, 1), dtype=dtype)

    def get_geo(self):
        file = self.root_path / f'sentinel1_{self.site}_{self.t1}.tif'
        _, transform, crs = geofiles.read_tif(file)
        return transform, crs

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'