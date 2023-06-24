from typing import Callable, List, NoReturn, Tuple

from langchain.prompts import PromptTemplate
from torch.utils.data import Dataset

from data.data import read_data


class RedditDataset(Dataset):
    def __init__(self,
                 path_to_data: str,
                 source_fname: str,
                 target_fname: str,
                 prompt: PromptTemplate,
                 preprocess_func: Callable = None,
                 ) -> NoReturn:
        super().__init__()
        self.source = read_data(path_to_data, source_fname)
        self.target = read_data(path_to_data, target_fname)
        self.prompt = prompt

        self.preprocess_func = preprocess_func
        self.source_preprocessed = None

    def preprocess(self, **kwargs) -> NoReturn:
        assert self.preprocess_func is not None, "Preprocess function not defined"
        self.source_preprocessed = [self.preprocess_func(entry, **kwargs) for entry in self.source]

    def __getitem__(self, idx: int) -> Tuple[List[str], List[str]]:
        if self.source_preprocessed is not None:
            return self.prompt.format(text=self.source_preprocessed[idx]), self.target[idx].strip('\n')
        return self.prompt.format(text=self.source[idx]), self.target[idx].strip('\n')

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
