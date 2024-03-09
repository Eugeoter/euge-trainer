import torch
import json
from pathlib import Path
from .sdxl_dataset_utils import get_input_ids


class DistillationDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer1, tokenizer2, proc_func):
        self.metadata_file = Path(args.metadata_file).absolute()
        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2
        self.max_token_length = args.max_token_length
        self.batch_size = args.batch_size
        self.proc_func = proc_func

        with open(self.metadata_file, "r") as f:
            self.metadata = json.load(f)

        prompts = []
        for img_key, img_md in self.metadata.items():
            caption = img_md.get('tags', None) or img_md.get('caption', None)
            if caption is None:
                continue
            prompts.append(caption)

        self.batches = []
        self.make_batches()

    def make_batches(self):
        batches = []
        for i in range(0, len(self.prompts), self.batch_size):
            batches.append(self.prompts[i:i + self.batch_size])
        self.batches = batches

    def __getitem__(self, index):
        batch = self.batches[index]
        sample = dict(
            student_prompts=[],
            student_input_ids_1=[],
            student_input_ids_2=[],
            teacher_prompts=[],
            teacher_input_ids_1=[],
            teacher_input_ids_2=[],
        )
        for prompt in batch:
            student_prompt = prompt
            student_input_ids_1 = get_input_ids(student_prompt, self.tokenizer1, max_token_length=self.max_token_length)
            student_input_ids_2 = get_input_ids(student_prompt, self.tokenizer2, max_token_length=self.max_token_length)
            teacher_prompt = self.proc_func(student_prompt)
            teacher_input_ids_1 = get_input_ids(teacher_prompt, self.tokenizer1, max_token_length=self.max_token_length)
            teacher_input_ids_2 = get_input_ids(teacher_prompt, self.tokenizer2, max_token_length=self.max_token_length)
            sample["student_prompts"].append(student_prompt)
            sample["student_input_ids_1"].append(student_input_ids_1)
            sample["student_input_ids_2"].append(student_input_ids_2)
            sample["teacher_prompts"].append(teacher_prompt)
            sample["teacher_input_ids_1"].append(teacher_input_ids_1)
            sample["teacher_input_ids_2"].append(teacher_input_ids_2)

        sample["student_input_ids_1"] = torch.stack(sample["student_input_ids_1"], dim=0)
        sample["student_input_ids_2"] = torch.stack(sample["student_input_ids_2"], dim=0)
        sample["teacher_input_ids_1"] = torch.stack(sample["teacher_input_ids_1"], dim=0)
        sample["teacher_input_ids_2"] = torch.stack(sample["teacher_input_ids_2"], dim=0)
        return sample
