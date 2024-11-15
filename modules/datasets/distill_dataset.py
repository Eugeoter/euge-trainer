from typing import Callable, List, Dict, Any
from .t2i_dataset import T2IDataset


class DistillDataset(T2IDataset):
    teacher_caption_getter: Callable[[dict], str] = lambda self, img_md, *args, **kwargs: img_md.get('caption') or ''
    teacher_negative_caption_getter: Callable[[dict], str] = lambda self, img_md, *args, **kwargs: img_md.get('negative_caption') or ''

    def get_teacher_caption(self, img_md, is_flipped=False, student_caption=None):
        return self.teacher_caption_getter(img_md, dataset_hook=self.dataset_hook, is_flipped=is_flipped, student_caption=student_caption)

    def get_teacher_negative_caption(self, img_md, is_flipped=False, student_negative_caption=None):
        return self.teacher_negative_caption_getter(img_md, dataset_hook=self.dataset_hook, is_flipped=is_flipped, student_caption=student_negative_caption)

    def get_teacher_caption_sample(self, batch: List[str], samples: Dict[str, Any]) -> Dict:
        sample = dict(
            teacher_captions=[],
        )
        for i, img_key in enumerate(batch):
            img_md = self.dataset[img_key]
            student_caption = samples['captions'][i]
            is_flipped = samples['is_flipped'][i]
            teacher_caption = self.get_teacher_caption(img_md, is_flipped=is_flipped, student_caption=student_caption)
            sample['teacher_captions'].append(teacher_caption)
        return sample
