import random

from typing import Callable, List, NoReturn, Tuple

from langchain.prompts import PromptTemplate
from torch.utils.data import Dataset

from data.data import read_data


class RedditDataset(Dataset):
    def __init__(self,
                 path_to_data: str,
                 source_fname: str,
                 target_fname: str,
                 preprocess_func: Callable,
                 prompt: PromptTemplate = None,
                 type_: str = 'train',
                 prompting_type: str = 'zero-shot',
                 prompt_dataset: 'RedditDataset' = None,
                 ) -> NoReturn:
        super().__init__()
        self.type_ = type_
        self.source = read_data(path_to_data, source_fname)
        self.target = read_data(path_to_data, target_fname)
        self.prompt = prompt
        self.prompting_type = prompting_type
        self.prompt_dataset = prompt_dataset

        self.preprocess_func = preprocess_func
        self.source_preprocessed = None

    def preprocess(self, **kwargs) -> NoReturn:
        self.source_preprocessed = [self.preprocess_func(entry, **kwargs) for entry in self.source]

    def __getitem__(self, idx: int) -> Tuple[List[str], List[str]]:
        assert self.prompting_type in ['zero-shot', 'one-shot', 'two-shot'], "Unknown prompting type"
        if self.prompting_type != 'zero-shot':
            assert self.prompt_dataset is not None, "Dataset for drawing prompts not defined"
        if self.prompting_type == 'zero-shot':
            if self.prompt is not None:
                return self.prompt.format(text=self.source_preprocessed[idx]), self.target[idx].strip('\n')
            return self.source_preprocessed[idx], self.target[idx].strip('\n')

        elif self.prompting_type == 'one-shot':
            sample_idx = random.choice(list(range(len(self.source))))
            if self.type_ == 'train': # avoid sampling the same example
                while sample_idx == idx:
                    sample_idx = random.choice(list(range(len(self.source))))
            sample_source, sample_target = self.prompt_dataset[sample_idx]
            return self.prompt.format(sample_source=sample_source,
                                      sample_target=sample_target,
                                      text=self.source_preprocessed[idx]), self.target[idx].strip('\n')

        else:  # two shot
            sample_idx = random.sample(list(range(len(self.source))), 2)
            if self.type_ == 'train': # avoid sampling the same example
                while idx in sample_idx:
                    sample_idx = random.sample(list(range(len(self.source))), 2)
            sample_source0, sample_target0 = self.prompt_dataset[sample_idx[0]]
            sample_source1, sample_target1 = self.prompt_dataset[sample_idx[1]]
            return self.prompt.format(sample_source0=sample_source0,
                                      sample_target0=sample_target0,
                                      sample_source1=sample_source1,
                                      sample_target1=sample_target1,
                                      text=self.source_preprocessed[idx]), self.target[idx].strip('\n')

    def __len__(self) -> int:
        return len(self.source)


def preprocess(text: str,
               return_title: bool = False,
               join_str: str = '\n') -> str:
    title = text.strip().split('   ')[0]
    raw_text = ' '.join(text.split('   ')[2:])
    raw_text = [lines.strip(' \n') for lines in raw_text.split('</s>')]
    output = [f'{lines}' for i, lines in enumerate(raw_text[1:], 1)]
    if return_title:
        title = title.split(': ')[1]
        output = [f'Original poster: {title} {raw_text[0]}'] + output
    else:
        output = [f'Original poster: {raw_text[0]}'] + output
    return join_str.join(output)
