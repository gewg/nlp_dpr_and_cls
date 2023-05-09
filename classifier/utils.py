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

label_to_id = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2, "DISPUTED": 3}

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

def output_file(filepath, output_data):
    f = open(filepath, "w")
    json.dump(output_data, f)
    f.close()

class ClaimDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 filepath_claims,
                 filepath_evidences):
    
        # load data
        with open(filepath_claims, 'r') as f:
            self.data_claims = json.load(f)
        with open(filepath_evidences, 'r') as f:
            self.data_evidences = json.load(f)

        # get attributes from data
        self.claims_ids = list(self.data_claims.keys())
        self.num_claims = len(self.claims_ids)
        self.tokenizer = tokenizer
    
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
        curr_claim_label = curr_claim["claim_label"]
        curr_claim_evidences_ids = curr_claim["evidences"]

        # preprocess the data
        curr_claim_text = self.data_preprocess(curr_claim_text)

        # output the data
        return (curr_claim_idx, curr_claim_text, curr_claim_label, curr_claim_evidences_ids)

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
        claims_ids = []
        claims_labels = []

        # load data from tuples
        for curr_claim_idx, curr_claim_text, curr_claim_label, curr_claim_evidences_ids in claims_tuples:

            claims_ids.append(curr_claim_idx)
            claims_labels.append(label_to_id[curr_claim_label])

            # combine the evidence's text with claim text
            curr_claim_evidence_text = ""
            curr_claim_evidence_text += curr_claim_text
            # add evidence's text, separate the evidences by separation token
            for curr_evidence_id in curr_claim_evidences_ids:
                curr_claim_evidence_text += self.tokenizer.sep_token
                curr_claim_evidence_text += self.data_preprocess(self.data_evidences[curr_evidence_id])
            claims_texts.append(curr_claim_evidence_text)
        
        # tokenize the texts
        tok_claims_texts = self.tokenizer(
            claims_texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # generate one batch
        batch = dict()
        batch["claims_texts_input_ids"] = tok_claims_texts.input_ids
        batch["claims_texts_attention_mask"] = tok_claims_texts.attention_mask
        batch["claims_ids"] = claims_ids
        batch["claims_labels"] = claims_labels

        return batch