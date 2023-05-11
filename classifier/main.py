"""
Main function of claim classifier. The Dense Passage Retrieval is used.

@ Author: Wei Ge
@ Student Number: 1074198
"""
import numpy as np
import json
import transformers
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import wandb
from tqdm import tqdm
import utils
import models
import time
import datetime
import os
import argparse
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from transformers import get_linear_schedule_with_warmup

id_to_label = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT_ENOUGH_INFO", 3: "DISPUTED"}

retrain = None

def driver_predict():
    """
    Predict the evidences
    """
    # load parameters from configuration file
    params = utils.get_config_from_json("evidence_retriever/config.json")

    # set the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(params.tokenizer)

    # initialize the dataset
    test_dataset = utils.TestDataset(tokenizer, "result/test-claims-predictions.json", "data/evidence.json")

    # initialize the dataloader
    dataloader_test = DataLoader(test_dataset, batch_size=params.batch_size_test, shuffle=False, collate_fn=test_dataset.collate_fn)

    # create the encoder and load the set up
    pretrained_model = AutoModel.from_pretrained(params.tokenizer)
    cls_model = models.ClaimClassifier(pretrained_model)
    cls_model.load_state_dict(torch.load("classifier/model_states/cls_model.bin"))
    cls_model.cuda()
    
    # prediction
    print("Predict - Start predicting")
    prediction = predict_model(cls_model, dataloader_test)
    # store the prediction
    print("Predict - Output the file")
    output_claims = {}
    for batch in dataloader_test:
        for idx, claim_id in enumerate(batch["claims_ids"]):
            curr_output_claim = {}
            curr_output_claim["claim_text"] = batch["claims_texts"][idx]
            curr_output_claim["claim_label"] = prediction[claim_id]
            curr_output_claim["evidences"] = batch["claims_evidences_ids"][idx]
            output_claims[claim_id] = curr_output_claim
    
    utils.output_file("result/test-claims-predictions.json", output_claims)

    print("Predict - Finish Prediction")

def driver_train():
    """
    Train the model to classify claims
    """

    '''Prepare before training'''
    # load parameters from configuration file
    params = utils.get_config_from_json("classifier/config.json")

    # set the seed number 
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    np.random.seed(params.seed)

    # set the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(params.tokenizer)

    # initialize the dataset
    train_dataset = utils.ClaimDataset(tokenizer, "data/train-claims.json", "data/evidence.json")
    validate_dataset = utils.ClaimDataset(tokenizer, "data/dev-claims.json", "data/evidence.json")

    # initalize the dataloader
    dataloader_train = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=False, collate_fn=train_dataset.collate_fn)
    dataloader_validate = DataLoader(validate_dataset, batch_size=params.batch_size, shuffle=False, collate_fn=validate_dataset.collate_fn)

    # create classifier
    pretrained_model = AutoModel.from_pretrained(params.tokenizer)
    cls_model = models.ClaimClassifier(pretrained_model)
    if retrain:
        cls_model.load_state_dict(torch.load("classifier/model_states/cls_model.bin"))
    cls_model.cuda()

    # loss function
    loss_fn =  nn.CrossEntropyLoss()

    # create the optimizer
    optimizer = optim.Adam(cls_model.parameters(), lr=params.learning_rate)

    # create the learning rate scheduler
    total_steps = len(dataloader_train) * params.num_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=params.num_warmup_steps, num_training_steps=total_steps)

    '''Start training'''
    for epoch_i in range(0, params.num_epoch):

        print("")
        print("======== Epoch %d / %d ========" % (epoch_i+1, params.num_epoch))

        train_per_epoch(dataloader_train, dataloader_validate, cls_model, optimizer, scheduler, loss_fn,
          params.grad_norm,
          params.num_itr_validate,
          params.save_model_dir,
          epoch_i+1)
        
        print("")


def train_per_epoch(dataloader_train, dataloader_validate,
          cls_model,
          optimizer,
          scheduler,
          loss_fn,
          grad_norm,
          num_itr_validate,
          save_model_dir,                
          epoch_i):
    '''
    According to Dense Passage Retrival, train the encoders
    '''
    # count the training time
    t0 = time.time()

    # # set the model as training
    # encoder.train()

    # reset the gradient
    optimizer.zero_grad()

    # total loss
    total_train_loss = 0
    
    # best accuracy
    best_accuracy = 0

    for step, batch in enumerate(dataloader_train):
        print("Train - Start training")

        # move data to cuda
        utils.move_data_to_cuda(batch)

        # get the labels' probabilities from model
        pro_labels = cls_model(batch["claims_texts_input_ids"], batch["claims_texts_attention_mask"])

        # calculate the loss
        loss = loss_fn(pro_labels, torch.LongTensor(batch["claims_labels"]).cuda())
        loss.backward()

        # count the total loss
        total_train_loss += loss.item()

        # gradient clipping
        nn.utils.clip_grad_norm_(cls_model.parameters(), grad_norm)

        # update the parameters
        optimizer.step()

        # update the learning rate
        scheduler.step()

        # reset the gradient
        optimizer.zero_grad()

        print("Train - Finish training")
        print("")

        # validate and report the information per 'num_itr_validate' times iteration
        if step % num_itr_validate == 0 and not step == 0:

            # validate the model
            print("==================================================")
            print("Validate - Start validating")

            accuracy = validate_model(cls_model, dataloader_validate)

            print("Validate - Iteration %d" % step)
            print("Validate - The accuracy is: %.3f" % accuracy)
            print("==================================================")
            print("")

            # if the f-score is better, save the model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(cls_model.state_dict(), os.path.join(save_model_dir, "%.3f_cls_model.bin" % accuracy))
                print("==================================================")
                print("Save Model - Iteration %d" % step)
                print("Save Model - The best accuracy is: %.3f" % accuracy)
                print("==================================================")

    # finish one epoch, make summary
    avg_train_loss = total_train_loss / len(dataloader_train)
    training_time = utils.format_time(time.time() - t0)
    print("Finish Epoch - Epoch %d" % epoch_i)
    print("Finish Epoch - The best f_score is: %.3f" % best_accuracy)
    print("Finish Epoch - The average loss is: %.3f" % avg_train_loss)
    print("Finish Epoch - The training time is: {:}".format(training_time))


def predict_model(model, dataloader_claims):
    """
    According to the claim text, get the label

    :return: evidences (list)
    """

    # predicted labels
    prediction_result = dict()

    # predict all claim text
    print("Predict - Start predicting claims' labels")
    for batch in dataloader_claims:
        # move data to cuda
        utils.move_data_to_cuda(batch)
        # get the probabilities for each label, and select the larget probability
        pro_labels = model(batch["claims_texts_input_ids"], batch["claims_texts_attention_mask"]).argmax(-1).tolist()
        # assign label to claim
        for idx, claim_id in enumerate(batch["claims_ids"]):
            prediction_result[claim_id] = id_to_label[pro_labels[idx]] # transfer id to label

    return prediction_result


def validate_model(model, dataloader_validate):
    """
    Validate the model
    """

    correct_prediction = 0
    num_claims = 0

    # predicte the validating data
    prediction = predict_model(model, dataloader_validate)

    # compare the prediction with the real result
    for batch in tqdm(dataloader_validate):
        # move data to cuda
        utils.move_data_to_cuda(batch)
        for idx, claim_id in enumerate(batch["claims_ids"]):
            num_claims += 1
            # transfer the label_id into label, then compare
            if id_to_label[batch["claims_labels"][idx]] == prediction[claim_id]:
                correct_prediction += 1
    
    # calculate the accuracy
    accuracy = correct_prediction / num_claims
    
    return accuracy


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="parms")
    parser.add_argument("--predict", action="store_true", help="predict the labels")
    parser.add_argument("--train", action="store_true", help="train a new model")
    parser.add_argument("--retrain", action="store_true", help="train the previous model")
    args = parser.parse_args()

    if args.predict:
        print("Starting...")
        driver_predict()
    if args.train:
        print("Starting...")
        retrain = False
        driver_train()
    if args.retrain:
        print("Starting...")
        retrain = True
        driver_train()