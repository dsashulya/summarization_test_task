import os

from typing import List, NoReturn


def read_data(path_to_data: str, fname: str) -> List[str]:
    data = []
    with open(os.path.join(path_to_data, fname)) as file:
        for line in file:
            if len(line) > 10:
                if 'source' in fname:
                    if line.startswith('Title:'):
                        data.append(line)
                    else:
                        data[-1] = data[-1].strip() + ' ' + line.strip()
                elif 'target' in fname:
                    data.append(line)
    return data


def write_data(path: str, data: List[str]) -> NoReturn:
    with open(path, 'w') as file:
        for line in data:
            file.write(line.strip() + '\n')
