"""
Analyse the project's training data

@ Author: Wei Ge
@ Student Number: 1074198
"""

import json

def data_analysis_claims(filepath):
    '''
    Get length of evidence and claim-text in each claim. Then calculate the maximum, minimum and average value.

    : param filepath: the path of file, it can be "train-claims.json" or "dev-claims.json"
    '''
    # load the data
    with open(filepath, 'r') as f:
        dataset = json.load(f)
    # collect the evidence and claim's length
    evidences_length = []
    claims_length = []
    for value in dataset.values():
        evidences_length.append(len(value["evidences"]))
        claims_length.append(len(value["claim_text"].split()))
    # calculate the maximum, minimum and average length
    print("Number of training data: %d" % len(dataset))
    print("Claim-evidences' length - Max:%d, Min:%d, Mean:%2f" % (max(evidences_length), min(evidences_length), sum(evidences_length) / len(evidences_length)))
    print("Claim-texts' length - Max:%d, Min:%d, Mean:%2f" % (max(claims_length), min(claims_length), sum(claims_length) / len(claims_length)))

def data_analysis_evidences(filepath):
    '''
    Get length of each evidence. Then calculate the maximum, minimum and average value.

    : param filepath: the path of file, it can be "evidence.json"
    '''
    # load the data
    with open(filepath, 'r') as f:
        dataset = json.load(f)
    # collect the evidence's length
    evidences_length = []
    for value in dataset.values():
        evidences_length.append(len(value))
    # calculate the maximum, minimum and average length
    print("Number of evidences - %d" % len(dataset))
    print("Evidences' length - Max:%d, Min:%d, Mean:%2f" % (max(evidences_length), min(evidences_length), sum(evidences_length) / len(evidences_length)))


# show the information
print("train-claims.json:")
data_analysis_claims("data/train-claims.json")
print("")
print("dev-claims.json:")
data_analysis_claims("data/dev-claims.json")
print("")
print("evidence.json:")
data_analysis_evidences("data/evidence.json")