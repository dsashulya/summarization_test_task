import json
import os

from datetime import datetime
from collections import defaultdict
from typing import NoReturn

import evaluate
import torch
from accelerate import init_empty_weights
from langchain.prompts import PromptTemplate
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LongT5ForConditionalGeneration,
)

from data.dataset import RedditDataset, preprocess, preprocess_arg_filtered

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else torch.device('cpu')
MODELS = {
    "LongT5": LongT5ForConditionalGeneration,
}


def compute_metrics(dataset: DataLoader,
                    model: AutoModelForCausalLM,
                    tokenizer: AutoTokenizer,
                    metric,
                    prompt: PromptTemplate = None,
                    logdir: str = None,
                    data_type: str = 'val',
                    iterations: int = 1,
                    summ_kwargs: dict = None) -> dict:
    output_metrics = defaultdict(float)
    final_max_length = summ_kwargs['max_length']
    with torch.no_grad():
        for texts, summaries in tqdm(dataset,
                                     position=0,
                                     leave=True):
            inputs = tokenizer(texts,
                               padding='longest',
                               return_tensors='pt')
            summ_kwargs['max_length'] = final_max_length * iterations
            for iteration in range(iterations):
                preds = model.generate(inputs['input_ids'].to(DEVICE), **summ_kwargs)
                preds = tokenizer.batch_decode(preds.cpu(),
                                               skip_special_tokens=True)
                inputs = [prompt.format(text=pred) for pred in preds]
                inputs = tokenizer(inputs,
                                   padding='longest',
                                   return_tensors='pt')
                summ_kwargs['max_length'] -= final_max_length

            metric.add_batch(predictions=preds, references=summaries)

            if logdir is not None:
                with open(os.path.join(logdir, f'{data_type}_generated_summaries.txt'), 'ab') as file:
                    for pred in preds:
                        file.write((pred + '\n\n').encode('utf-8'))
        summ_kwargs['max_length'] = final_max_length
        return metric.compute()


def main() -> NoReturn:
    # parameters
    with open('params/prompt.txt', 'r') as prompt_file:
        prompt_text = prompt_file.readline().strip()

    with open('params/data_params.json', 'r') as data_params_file:
        data_params = json.load(data_params_file)

    with open('params/validation_params.json', 'r') as validation_params_file:
        validation_params = json.load(validation_params_file)

    # logging
    os.makedirs(validation_params['log']['logdir'], exist_ok=True)
    now = datetime.now()
    if validation_params['log']['folder_name'] is not None:
        log_dir = os.path.join(validation_params['log']['logdir'],
                               validation_params['log']['folder_name'])
    else:
        log_dir = os.path.join(validation_params['log']['logdir'],
                               f'{validation_params["model"]["model_name"]}-{now:%Y%m%d-%H%M-%S}')
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, 'prompt.txt'), 'w') as file:
        file.write(prompt_text)
    with open(os.path.join(log_dir, 'data_params.json'), 'w') as file:
        json.dump(data_params, file)
    with open(os.path.join(log_dir, 'validation_params.json'), 'w') as file:
        json.dump(validation_params, file)

    # data
    if validation_params['prompting'] == 'zero-shot':
        inp_vars = ["text"]
    elif validation_params['prompting'] == 'one-shot':
        inp_vars = ["sample_source", "sample_target", "text"]
    else:
        inp_vars = ["sample_source0", "sample_target0",
                    "sample_source1", "sample_target1",
                    "text"]

    prompt = PromptTemplate(
        input_variables=inp_vars,
        template=prompt_text,
    )

    preprocess_func = preprocess if data_params['path_to_data'].split('/')[-1] == 'vanilla' else preprocess_arg_filtered
    if validation_params['prompting'] != 'zero-shot':
        prompt_ds = RedditDataset(data_params['paths']['path_to_data'],
                                  data_params['paths']['train_source'],
                                  data_params['paths']['train_target'],
                                  prompt=None,
                                  preprocess_func=preprocess_func,
                                  type_='train',
                                  prompting_type='zero-shot')
        prompt_ds.preprocess(**data_params['preprocess_kwargs'])
    else:
        prompt_ds = None

    train = RedditDataset(data_params['paths']['path_to_data'],
                          data_params['paths']['train_source'],
                          data_params['paths']['train_target'],
                          prompt=prompt,
                          preprocess_func=preprocess_func,
                          type_='train',
                          prompting_type=validation_params['prompting'],
                          prompt_dataset=prompt_ds)
    val = RedditDataset(data_params['paths']['path_to_data'],
                        data_params['paths']['val_source'],
                        data_params['paths']['val_target'],
                        prompt=prompt,
                        preprocess_func=preprocess_func,
                        type_='val',
                        prompting_type=validation_params['prompting'],
                        prompt_dataset=prompt_ds)
    test = RedditDataset(data_params['paths']['path_to_data'],
                         data_params['paths']['test_source'],
                         data_params['paths']['test_target'],
                         prompt=prompt,
                         preprocess_func=preprocess_func,
                         type_='test',
                         prompting_type=validation_params['prompting'],
                         prompt_dataset=prompt_ds)

    train.preprocess(**data_params['preprocess_kwargs'])
    val.preprocess(**data_params['preprocess_kwargs'])
    test.preprocess(**data_params['preprocess_kwargs'])

    train_dl = DataLoader(train, **validation_params['dataloader'])
    val_dl = DataLoader(val, **validation_params['dataloader'])
    test_dl = DataLoader(test, **validation_params['dataloader'])

    # model
    config = AutoConfig.from_pretrained(validation_params['model']['model_path'])
    with init_empty_weights():
        model = MODELS[validation_params['model']['model_name']](config)
        model.tie_weights()
    model = model.from_pretrained(validation_params['model']['model_path'],
                                  device_map='auto',
                                  torch_dtype='auto',
                                  )
    tokenizer = AutoTokenizer.from_pretrained(validation_params['model']['model_path'],
                                              use_fast=True)

    # metrics
    rouge = evaluate.load("rouge")

    # validation
    metrics_train = compute_metrics(train_dl,
                                    model,
                                    tokenizer,
                                    rouge,
                                    iterations=validation_params['iterations'],
                                    prompt=prompt,
                                    summ_kwargs=validation_params["generate"])
    metrics_val = compute_metrics(val_dl,
                                  model,
                                  tokenizer,
                                  rouge,
                                  iterations=validation_params['iterations'],
                                  prompt=prompt,
                                  logdir=log_dir,
                                  data_type='val',
                                  summ_kwargs=validation_params["generate"])
    metrics_test = compute_metrics(test_dl,
                                   model,
                                   tokenizer,
                                   rouge,
                                   iterations=validation_params['iterations'],
                                   prompt=prompt,
                                   logdir=log_dir,
                                   data_type='test',
                                   summ_kwargs=validation_params["generate"])

    with open(os.path.join(log_dir, 'metrics_train.txt'), 'w') as file:
        for name, value in metrics_train.items():
            file.write(f'{name}: {str(value)}\n')
    with open(os.path.join(log_dir, 'metrics_val.txt'), 'w') as file:
        for name, value in metrics_val.items():
            file.write(f'{name}: {str(value)}\n')
    with open(os.path.join(log_dir, 'metrics_test.txt'), 'w') as file:
        for name, value in metrics_test.items():
            file.write(f'{name}: {str(value)}\n')


if __name__ == '__main__':
    main()
