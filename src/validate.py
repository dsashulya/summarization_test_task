import json

from collections import defaultdict
from typing import NoReturn

import evaluate
import torch

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from langchain.prompts import PromptTemplate
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LongT5ForConditionalGeneration
)

from data.dataset import RedditDataset, preprocess


DEVICE = torch.device(f'cuda:{torch.cuda.get_device_name()}') if torch.cuda.is_available() else torch.device('cpu')
MODELS = {
    "LongT5": LongT5ForConditionalGeneration,
}


def compute_metrics(dataset: RedditDataset,
                    model: AutoModelForCausalLM,
                    tokenizer: AutoTokenizer,
                    metric,
                    summ_kwargs: dict = {}) -> dict:
    output_metrics = defaultdict(float)
    with torch.no_grad():
        for texts, summaries in tqdm(dataset,
                          position=0,
                          leave=True):
            inputs = tokenizer(texts,
                               padding='longest',
                               return_tensors='pt')
            preds = model.generate(inputs['input_ids'].to(DEVICE), **summ_kwargs)
            preds = tokenizer.batch_decode(preds.cpu(),
                                     skip_special_tokens=True)
            metric.add_batch(predictions=preds, references=summaries)
        return metric.compute()


def main() -> NoReturn:
    # parameters
    with open('params/prompt.txt', 'r') as prompt_file:
        prompt_text = prompt_file.readline().strip()

    with open('params/data_params.json', 'r') as data_params_file:
        data_params = json.load(data_params_file)

    with open('params/validation_params.json', 'r') as validation_params_file:
        validation_params = json.load(validation_params_file)
    
    # data
    prompt = PromptTemplate(
        input_variables=["text"],
        template=prompt_text,
    )

    train = RedditDataset(data_params['paths']['path_to_data'],
                          data_params['paths']['train_source'],
                          data_params['paths']['train_target'],
                          prompt,
                          preprocess_func=preprocess)
    val = RedditDataset(data_params['paths']['path_to_data'],
                          data_params['paths']['val_source'],
                          data_params['paths']['val_target'],
                          prompt,
                          preprocess_func=preprocess)
    test = RedditDataset(data_params['paths']['path_to_data'],
                          data_params['paths']['test_source'],
                          data_params['paths']['test_target'],
                          prompt,
                          preprocess_func=preprocess)

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
    metrics = compute_metrics(train_dl,
                              model,
                              tokenizer,
                              rouge,
                              summ_kwargs=validation_params["generate"])

if __name__ == '__main__':
    main()