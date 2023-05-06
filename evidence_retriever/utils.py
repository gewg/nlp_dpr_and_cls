"""
Utils and Dataset will be used

@ Author: Wei Ge
@ Student Number: 1074198
"""
import json
import random
import time
import datetime
from bunch import Bunch
from torch.utils.data import Dataset

def get_config_from_json(filepath):
    """
    Get the config from a json file

    :param json_file:
    :return: config(namespace)
    """
    with open(filepath, 'r') as config_file:
        config_dict = json.load(config_file)
    config = Bunch(config_dict)
    return config

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def move_data_to_cuda(batch):
    """
    Move data to cuda device
    """
    for key in batch.keys():
        if key in ["claims_texts_input_ids", "claims_texts_attention_mask",
                   "evidences_input_ids", "evidences_attention_mask"]:
                   batch[key] = batch[key].cuda()

def get_fscore(precision, recall):
    return 2 * (precision * recall) / (precision + recall)

class ClaimDataset(Dataset):
    def __init__(self, num_evidence_per_batch, 
                 tokenizer,
                 filepath_claims, filepath_evidences):
        
        # initalize variables
        self.num_evidence_per_batch = num_evidence_per_batch
        self.tokenizer = tokenizer
    
        # load data
        with open(filepath_claims, 'r') as f:
            self.data_claims = json.load(f)
        with open(filepath_evidences, 'r') as f:
            self.data_evidences = json.load(f)

        # get attributes from data
        self.claims_ids = list(self.data_claims.keys())
        self.num_claims = len(self.claims_ids)
        self.evidences_ids = list(self.data_evidences.keys())
        self.num_evidences = len(self.evidences_ids)
    
    def __len__(self):
        '''
        Get length of the dataset
        '''
        return self.num_claims
    
    def __getitem__(self, idx):
        '''
        Output the data for training
        '''
        # get the claim's data according to the index
        curr_claim_idx = self.claims_ids[idx]
        curr_claim = self.data_claims[curr_claim_idx]
        curr_claim_text = curr_claim["claim_text"]
        curr_claim_evidences = curr_claim["evidences"]

        # preprocess the data
        curr_claim_text = self.data_preprocess(curr_claim_text)

        # output the data
        return (curr_claim_text, curr_claim_evidences)

    def data_preprocess(self, data):
        '''
        Preprocess the data
        '''
        data = data.lower()
        return data

    def collate_fn(self, claims_tuples):
        """
        Process the inputed list of claims' data, output one batch

        :param claims_tuples: list of tuples. [ (claim_text, claim_evidence), (claim_text, claim_evidence), ...]
        :return: one batch (dictinaory)
        """
        claims_texts = []
        claims_evidences_ids = []
        claims_evidences_texts = []
        claims_positive_evidences_positions = []

        # load data from tuples
        for curr_claim_text, curr_claim_evidences_ids in claims_tuples:
            # mark the positive evidences' start and end position in evidences' list
            curr_positive_evidence_start = len(claims_evidences_ids)
            curr_positive_evidence_end = curr_positive_evidence_start + len(curr_claim_evidences_ids) - 1
            claims_positive_evidences_positions.append([curr_positive_evidence_start, curr_positive_evidence_end])
            
            claims_texts.append(curr_claim_text)
            claims_evidences_ids.extend(curr_claim_evidences_ids)
        
        # if the number of existed evidences is less than the expected size, randomly pick other evidences to fill the batch
        num_claims_evidences = len(claims_evidences_ids)
        while num_claims_evidences < self.num_evidence_per_batch:
            random_evidence_id = random.choice(self.evidences_ids)
            # avoid the duplicated evidences
            while random_evidence_id in claims_evidences_ids:
                random_evidence_id = random.choice(self.evidences_ids)
            # filling the evidences in batch
            claims_evidences_ids.append(random_evidence_id)
            num_claims_evidences += 1

        # according to evidences' ids, get evidences' texts
        for id in claims_evidences_ids:
            claims_evidences_texts.append(self.data_preprocess(self.data_evidences[id]))

        # tokenize the texts
        tok_claims_texts = self.tokenizer(
            claims_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        tok_claims_evidences_texts = self.tokenizer(
            claims_evidences_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # generate one batch
        batch = dict()
        batch["claims_texts_input_ids"] = tok_claims_texts.input_ids
        batch["claims_texts_attention_mask"] = tok_claims_texts.attention_mask
        batch["evidences_input_ids"] = tok_claims_evidences_texts.input_ids
        batch["evidences_attention_mask"] = tok_claims_evidences_texts.attention_mask
        batch["claims_positive_evidences_positions"] = claims_positive_evidences_positions

        return batch

class EvidenceDataset(Dataset):
    def __init__(self, tokenizer, filepath_evidences):
        
        # initalize variables
        self.tokenizer = tokenizer
    
        # load data
        with open(filepath_evidences, 'r') as f:
            self.data_evidences = json.load(f)

        # get attributes from data
        self.evidences_ids = list(self.data_evidences.keys())
        self.num_evidences = len(self.evidences_ids)
    
    def __len__(self):
        return self.num_evidences
    
    def __getitem__(self, idx):
        # get the claim's data according to the index
        curr_evidence_id = self.evidences_ids[idx]
        curr_evidence_text = self.data_evidences[curr_evidence_id]

        # preprocess the data
        curr_evidence_text = self.data_preprocess(curr_evidence_text)

        # output the data
        return (curr_evidence_id, curr_evidence_text)
    
    def data_preprocess(self, data):
        '''
        Preprocess the data
        '''
        data = data.lower()
        return data
    
    def collate_fn(self, evidences_tuples):
        """
        Process the inputed list of evidences' data, output one batch

        :param evidences_tuples: list of tuples. [ (evidence_id, evidence_text), (evidence_id, evidence_text), ...]
        :return: one batch (dictinaory)
        """
        evidences_ids = []
        evidences_texts = []

        # load data from tuples
        for curr_evidence_id, curr_evidence_text in evidences_tuples:
            evidences_texts.append(curr_evidence_text)
            evidences_ids.append(curr_evidence_id)

        # tokenize the texts
        tok_evidences_text = self.tokenizer(
            evidences_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # generate one batch
        batch = dict()
        batch["evidences_ids"] = evidences_ids
        batch["evidences_input_ids"] = tok_evidences_text.input_ids
        batch["evidences_attention_mask"] = tok_evidences_text.attention_mask

        return batch
    
class ValidateDataset(Dataset):
    def __init__(self, tokenizer, filepath_validate):
        # initalize variables
        self.tokenizer = tokenizer
    
        # load data
        with open(filepath_validate, 'r') as f:
            self.data_validate_claims = json.load(f)

        # get attributes from data
        self.validate_claims_ids = list(self.data_validate_claims.keys())
        self.num_validate_claims = len(self.validate_claims_ids)
    
    def __len__(self):
        return self.num_validate_claims
    
    def __getitem__(self, idx):
        # get the claim's data according to the index
        curr_validate_claim_id = self.validate_claims_ids[idx]
        curr_validate_claim = self.data_validate_claims[curr_validate_claim_id]
        curr_validate_claim_text = curr_validate_claim["claim_text"]
        curr_validate_claim_evidences_ids = curr_validate_claim["evidences"]

        # preprocess the data
        curr_validate_claim_text = self.data_preprocess(curr_validate_claim_text)

        # output the data
        return (curr_validate_claim_id, curr_validate_claim_text, curr_validate_claim_evidences_ids)
    
    def data_preprocess(self, data):
        '''
        Preprocess the data
        '''
        data = data.lower()
        return data
    
    def collate_fn(self, validation_tuples):
        """
        Process the inputed list of validation' data, output one batch

        :param evidences_tuples: list of tuples. [ (curr_validate_claim_id, curr_validate_claim_text, curr_validate_claim_evidences_ids), ...]
        :return: one batch (dictinaory)
        """

        validate_claim_ids = []
        validate_claim_texts = []
        validate_claim_evidences_ids = []

        # load data from tuples
        for curr_validate_claim_id, curr_validate_claim_text, curr_validate_claim_evidences_ids in validation_tuples:
            validate_claim_ids.append(curr_validate_claim_id)
            validate_claim_texts.append(curr_validate_claim_text)
            validate_claim_evidences_ids.append(curr_validate_claim_evidences_ids)

        # tokenize the texts
        tok_validate_claim_texts = self.tokenizer(
            validate_claim_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # generate one batch
        batch = dict()
        batch["claim_ids"] = validate_claim_ids
        batch["claim_evidences_ids"] = validate_claim_evidences_ids
        batch["claims_texts_input_ids"] = tok_validate_claim_texts.input_ids
        batch["claims_texts_attention_mask"] = tok_validate_claim_texts.attention_mask

        return batch


# class PredictDataset(Dataset):
#     def __init__(self, tokenizer, filepath_prediction):
#         # initalize variables
#         self.tokenizer = tokenizer
    
#         # load data
#         with open(filepath_prediction, 'r') as f:
#             self.data_prediction = json.load(f)

#         # get attributes from data
#         self.prediction_ids = list(self.data_prediction.keys())
#         self.num_prediction = len(self.prediction_ids)
    
#     def __len__(self):
#         return self.num_prediction
    
#     def __getitem__(self, idx):
#         # get the claim's data according to the index
#         curr_prediction_ids = self.prediction_ids[idx]
#         curr_prediction = self.data_prediction[curr_prediction_ids]

#         # preprocess the data
#         curr_prediction = self.data_preprocess(curr_prediction)

#         # output the data
#         return (curr_prediction_ids, curr_prediction)
    
#     def data_preprocess(self, data):
#         '''
#         Preprocess the data
#         '''
#         data = data.lower()
#         return data
    
#     def collate_fn(self, evidences_tuples):
#         """
#         Process the inputed list of evidences' data, output one batch

#         :param evidences_tuples: list of tuples. [ (evidence_id, evidence_text), (evidence_id, evidence_text), ...]
#         :return: one batch (dictinaory)
#         """
#         evidences_ids = []
#         evidences_texts = []