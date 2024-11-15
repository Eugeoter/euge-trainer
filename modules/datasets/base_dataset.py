import torch
import os
import time
import random
import functools
from PIL import Image
from waifuset import HuggingFaceData, DictDataset, FastDataset
from waifuset import logging
from waifuset import const
from ml_collections import ConfigDict
from accelerate import Accelerator
from typing import Any, Callable, Dict, List, Union, Literal
from ..utils import class_utils


def default_dataset_hook_saver(dataset_hook, path):
    r"""
    Default dataset_hook_saver for BaseDataset.
    """
    import json
    with open(path, 'w') as f:
        json.dump(dataset_hook, f, indent=4)


def custom_func(func):
    r"""
    Decorator for custom functions.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


class BaseDataset(torch.utils.data.Dataset, class_utils.FromConfigMixin):
    accelerator: Accelerator

    dataset_source: Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]
    dataset_merge_mode: Literal['union', 'intersection'] = 'union'

    max_retries: int = None  # infinite retries
    hf_cache_dir: str = None
    hf_token: str = None

    split: Literal['train', 'valid', 'test'] = 'train'

    batch_size: int = 1
    max_dataset_n_workers: int = 1

    output_dir: str
    output_subdir: ConfigDict

    record_columns: List[str] = None
    record_dataset_hook: bool = True

    image_key_getter: Callable[[Dict[str, Any]], str] = None
    image_getter: Callable[[Dict[str, Any]], Image.Image] = None

    data_preprocessor: Callable[[Dict[str, Any]], Dict[str, Any]] = None
    dataset_hook_getter: Callable[['BaseDataset'], Any] = None
    dataset_hook_saver: Callable[[Any, str], None] = lambda self, dataset_hook, path: default_dataset_hook_saver(dataset_hook, path)
    data_weight_getter: Callable[[Dict[str, Any]], int] = None

    @classmethod
    def from_config(cls, config, accelerator, **kwargs):
        return super().from_config(config, accelerator=accelerator, **kwargs)

    def check_config(self):
        pass

    def setup(self):
        tic_setup = time.time()
        timer = {}
        self._setup_basic()
        setups = self.get_setups()
        for setup in setups:
            tic = time.time()
            setup()
            timer[setup.__name__[7:]] = time.time() - tic

        self.logger.print(f"setup dataset for process {self.accelerator.local_process_index}/{self.accelerator.num_processes}: {time.time() - tic_setup:.2f}s", disable=False)
        for key, value in timer.items():
            self.logger.print(f"    {key}: {value:.2f}s", no_prefix=True)

    def get_setups(self):
        return [
            self._setup_basic,
            self._setup_dataset,
            self._setup_pre_dataset_hook,
            self._setup_data,
            self._setup_post_dataset_hook,
            self._setup_batches,
        ]

    def _setup_basic(self):
        self.logger = logging.get_logger("dataset", disable=not self.accelerator.is_main_process)
        self.check_config()

        self.samplers = self.get_samplers()
        self.record_dir = os.path.join(self.output_dir, self.split, self.output_subdir.records) if 'records' in self.output_subdir else None

    def _setup_dataset(self):
        if not hasattr(self, 'dataset'):
            self.dataset = self.load_dataset()
        self.logger.info("Loaded dataset:")
        self.logger.info(self.dataset, no_prefix=True)

        missing_cnt = 0
        for img_key, img_md in list(self.dataset.items()):
            if not self.get_data_existence(img_md):
                del self.dataset[img_key]
                missing_cnt += 1
        self.logger.info(f"number of missing images: {missing_cnt}, left: {len(self.dataset)}")

    def _setup_pre_dataset_hook(self):
        self.dataset_hook = self.get_dataset_hook()
        if self.record_dir is not None and self.record_columns is not None and self.record_dataset_hook and self.accelerator.is_main_process and self.dataset_hook_saver is not None:
            try:
                os.makedirs(self.record_dir, exist_ok=True)
                save_path = os.path.join(self.record_dir, 'dataset_hook_raw.json')
                self.dataset_hook_saver(self.dataset_hook, save_path)
                self.logger.print(f"saved dataset_hook to: {save_path}")
            except Exception as e:
                self.logger.error(f"failed to save dataset_hook: {e}")

    def _setup_data(self):
        # load data
        pbar = self.logger.tqdm(total=len(self.dataset), desc='load data')
        if self.max_dataset_n_workers <= 1:
            for img_md in self.dataset.values():
                self.load_data(img_md)
                pbar.update(1)
        else:
            import concurrent.futures
            pbar = self.logger.tqdm(total=len(self.dataset), desc='load data')
            load_data = logging.track_tqdm(pbar)(self.load_data)
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_dataset_n_workers) as executor:
                futures = {executor.submit(load_data, img_md): img_md for img_md in self.dataset.values()}
                for future in concurrent.futures.as_completed(futures):
                    future.result()
        pbar.close()

        # drop data with weight 0
        drops = 0
        for img_key, img_md in list(self.dataset.items()):
            if img_md.get('drop') or (img_md.get('weight', 1) <= 0):
                del self.dataset[img_key]
                drops += 1
        self.logger.print(f"number of drops: {drops}, left: {len(self.dataset)}")
        self.logger.print(f"number of repeats: {sum(img_md.get('weight', 1) for img_md in self.dataset.values())}")

    def _setup_post_dataset_hook(self):
        try:
            if self.record_dir is not None and self.record_columns is not None and self.record_dataset_hook and self.accelerator.is_main_process and self.dataset_hook_saver is not None:
                os.makedirs(self.record_dir, exist_ok=True)
                # save as csv
                import csv
                records = self.get_records()
                record_path = os.path.join(self.record_dir, 'records.csv')
                with open(record_path, 'w') as f:
                    writer = csv.DictWriter(f, fieldnames=['image_key'] + self.record_columns)
                    writer.writeheader()
                    for record in records.values():
                        writer.writerow(record)
                self.logger.print(f"recorded to: {record_path}")

        except Exception as e:
            self.logger.error(f"failed to save records: {e}")

        self.dataset_hook = self.get_dataset_hook()  # update dataset_hook
        if self.dataset_hook is not None and self.record_dataset_hook and self.accelerator.is_main_process:
            try:
                os.makedirs(self.record_dir, exist_ok=True)
                save_path = os.path.join(self.record_dir, 'dataset_hook.json')
                self.dataset_hook_saver(self.dataset_hook, save_path)
                self.logger.print(f"saved dataset_hook to: {save_path}")
            except Exception as e:
                self.logger.error(f"failed to save dataset_hook: {e}")

    def _setup_batches(self):
        self.logger.info("Making batches...")
        with self.logger.timer('Make batches'):
            self.batches = self.make_batches(self.dataset)
        self.logger.print(f"Total number of train batches: {len(self.batches)}")

    def get_custom_funcs(self):
        return [self.image_key_getter, self.image_getter, self.data_preprocessor, self.dataset_hook_getter, self.dataset_hook_saver, self.data_weight_getter]

    def load_data(self, img_md) -> Dict:
        img_md = self.get_preprocessed_img_md(img_md)
        extra_kwargs = {}
        weight = self.get_data_weight(img_md)
        if weight == 0:
            img_md.update(drop=True)
            return
        extra_kwargs.update(weight=weight)
        img_md.update(**extra_kwargs)

    def get_dataset_default_kwargs(self):
        return dict(
            cache_dir=self.hf_cache_dir,
            token=self.hf_token,
            max_retries=self.max_retries,
            fp_key='image_path',
            primary_key='image_key',
            tbname=None,
            recur=True,
            exts=const.IMAGE_EXTS,
            read_only=True,
            verbose=True,
        )

    def load_dataset(self) -> DictDataset:
        from waifuset.classes.dataset.fast_dataset import parse_source_input
        self.dataset_source = parse_source_input(self.dataset_source)
        default_kwargs = self.get_dataset_default_kwargs()
        dataset = FastDataset(self.dataset_source, dataset_cls=DictDataset, merge_mode=self.dataset_merge_mode, **default_kwargs)
        return dataset

    def get_data_existence(self, img_md):
        if isinstance(img_md, HuggingFaceData):
            return True
        return os.path.exists(img_md.get('image_path', '')) or os.path.exists(img_md.get('cache_path', ''))

    def get_img_md(self, img_key):
        return self.dataset[img_key]

    def get_dataset_hook(self) -> Any:
        if self.dataset_hook_getter is None:
            return None
        else:
            self.logger.info("Getting dataset_hook...")
            with self.logger.timer('Get dataset_hook'):
                return self.dataset_hook_getter(self)

    def get_records(self):
        self.logger.info("Getting records...")
        with self.logger.timer('Get records'):
            records = {}
            for img_key, img_md in self.dataset.items():
                record = {}
                record['image_key'] = img_key
                for col in self.record_columns:
                    record[col] = img_md.get(col)
                records[img_key] = record
            return records

    def get_preprocessed_img_md(self, img_md) -> Dict:
        if self.data_preprocessor is None:
            return img_md
        else:
            return self.data_preprocessor(img_md, dataset_hook=self.dataset_hook)

    def get_data_weight(self, img_md):
        if self.data_weight_getter is None:
            return 1
        else:
            return self.data_weight_getter(img_md, dataset_hook=self.dataset_hook)

    def make_batches(self, dataset) -> List[List[str]]:
        img_keys = list(dataset.keys())
        batches = []
        for i in self.logger.tqdm(range(0, len(dataset), self.batch_size), desc='make batches'):
            batch = img_keys[i:i+self.batch_size]
            batches.append(batch)
        return batches

    def get_samplers(self):
        r"""
        Get samplers in order of execution.
        """
        samplers = [sampler for sampler in dir(self) if sampler.startswith('get_') and sampler.endswith('_sample') and callable(getattr(self, sampler))]
        samplers.sort()
        samplers = [getattr(self, sampler) for sampler in samplers]
        return samplers

    def get_samples(self, batch: List[str]):
        samples = {}
        for sampler in self.samplers:
            samples.update(sampler(batch, samples))
        return samples

    def __getitem__(self, i):
        batch = self.batches[i]
        sample = self.get_samples(batch)
        return sample

    def __len__(self):
        return len(self.batches)

    @staticmethod
    def collate_fn(batch):
        return batch[0]

    def shuffle(self):
        random.shuffle(self.batches)
