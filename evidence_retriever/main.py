"""
Main function of evidence retriever. The Dense Passage Retrieval is used.

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
import utils
import time
import datetime
import os
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import get_linear_schedule_with_warmup

def driver_train():
    """
    Train the model of Dense Passage Retriever
    """

    '''Prepare before training'''
    # load parameters from configuration file
    params = utils.get_config_from_json("evidence_retriever/config.json")

    # set the seed number 
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)

    # set the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(params.tokenizer)

    # initialize the dataset
    train_dataset = utils.ClaimDataset(params.num_evidence_per_batch, tokenizer, "data/train-claims.json", "data/evidence/json")
    evidence_dataset = utils.EvidenceDataset(tokenizer, "data/evidence.json")
    validate_dataset = utils.ValidateDataset(tokenizer, "data/dev-claims.json")

    # initialize the dataloader
    dataloader_train = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    dataloader_evidence = DataLoader(evidence_dataset, batch_size=params.batch_size, shuffle=True, collate_fn=evidence_dataset.collate_fn)
    dataloader_validate = DataLoader(validate_dataset, batch_size=params.batch_size, shuffle=True, collate_fn=validate_dataset.collate_fn)

    # create the encoders, there are two encoders for 'question' and 'passage' according to dense passage retrieval
    encoder_claim = AutoModelForMaskedLM.from_pretrained(params.tokenizer)
    encoder_evidence = AutoModelForMaskedLM.from_pretrained(params.tokenizer)
    encoder_claim.cuda()
    encoder_evidence.cuda()

    # create the optimizer
    optimizer_claim = encoder_claim.Adam(encoder_claim.parameters())
    optimizer_evidence = encoder_evidence.Adam(encoder_evidence.parameters())

    # create the learning rate scheduler
    total_steps = len(dataloader_train) * params.num_epoch
    scheduler_claim = get_linear_schedule_with_warmup(optimizer=optimizer_claim, num_warmup_steps=params.num_warmup_steps, num_training_steps=total_steps)
    scheduler_evidence = get_linear_schedule_with_warmup(optimizer=optimizer_evidence, num_warmup_steps=params.num_warmup_steps, num_training_steps=total_steps)


    '''Start training'''
    for epoch_i in range(0, params.num_epoch):

        print("")
        print("======== Epoch {:} / {:} ========" % (epoch_i + 1, params.num_epoch))

        train_per_epoch(dataloader_train, dataloader_evidence, dataloader_validate,
          encoder_claim, encoder_evidence,
          optimizer_claim, optimizer_evidence,
          scheduler_claim, scheduler_evidence,
          params.similarity_adjustment,
          params.grad_norm,
          params.num_itr_validate,
          params.topk,
          params.save_model_dir,
          epoch_i)
        
        print("")
        


def train_per_epoch(dataloader_train, dataloader_evidence, dataloader_validate,
          encoder_claim, encoder_evidence,
          optimizer_claim, optimizer_evidence,
          scheduler_claim, scheduler_evidence,
          similarity_adjustment,
          grad_norm,
          num_itr_validate,
          topk,
          save_model_dir,
          epoch_i
          ):
    '''
    According to Dense Passage Retrival, train the encoders
    '''
    # count the training time
    t0 = time.time()

    # set the model as training
    encoder_claim.train()
    encoder_evidence.train()

    # reset the gradient
    optimizer_claim.zero_grad()
    optimizer_evidence.zero_grad()

    # total loss
    total_train_loss = 0
    
    # best f-score
    best_f_score = 0

    for step, batch in enumerate(dataloader_train):
        # move data to cuda
        utils.move_data_to_cuda(batch)
        # encode the claim text(question) and evidence text(passage)
        claim_text_embeddings = encoder_claim(input_ids=batch["claims_texts_input_ids"], 
                                              attention_mask=batch["claims_texts_attention_mask"]).last_hidden_state
        evidence_text_embeddings = encoder_evidence(input_ids=batch["evidences_input_ids"], 
                                                    attention_mask=batch["evidences_attention_mask"]).last_hidden_state
        # normalize the embeddings
        claim_text_embeddings = nn.functional.normalize(claim_text_embeddings[:, 0, :])
        evidence_text_embeddings = nn.functional.normalize(evidence_text_embeddings[:, 0, :])

        # calcualte the loss, according to the formulas in dense passage retrieval
        # calculate the similarities between two embeddings
        sims = torch.mm(claim_text_embeddings.t(), evidence_text_embeddings)
        # calculate the loss for each evidence
        losses = - nn.functional.log_softmax(sims / similarity_adjustment, dim=1)  # while the difference between similarities may be small, the similarity_adjustment is used to increase it
        # for each claim text, get the positive evidence's loss
        each_claim_loss = []
        for idx, positive_evidences_positions in enumerate(batch["claims_positive_evidences_positions"]):
            curr_claim_pos_evidence_start = positive_evidences_positions[0]
            curr_claim_pos_evidence_end = positive_evidences_positions[1]
            curr_claim_losses = losses[idx, curr_claim_pos_evidence_start:curr_claim_pos_evidence_end]
            # while each claim has more than one positive evidence, so calculate the mean as the loss
            each_claim_loss.append(torch.mean(curr_claim_losses))

        # backward the loss
        each_claim_loss = torch.stack(each_claim_loss).mean
        each_claim_loss.backward()

        # count the loss
        total_train_loss += each_claim_loss.item()

        # gradient clipping
        nn.utils.clip_grad_norm_(encoder_claim.parameters, grad_norm)
        nn.utils.clip_grad_norm_(encoder_evidence.parameters, grad_norm)

        # update the parameters
        optimizer_claim.step()
        optimizer_evidence.step()

        # update the learning rate
        scheduler_claim.step()
        scheduler_evidence.step()

        del claim_text_embeddings, evidence_text_embeddings, sims, losses

        # validate and report the information per 'num_itr_validate' times iteration
        if step % num_itr_validate == 0 and not step == 0:
            # validate the model
            f_score = validate_model(encoder_claim, encoder_evidence, 
                                     dataloader_validate, dataloader_evidence, 
                                     topk)
            print("Validate - Iteration %d" % step)
            print("Validate - The f_score is: %.3f" % f_score)
            
            # if the f-score is better, save the model
            if f_score > best_f_score:
                best_f_score = f_score
                torch.save(encoder_claim.state_dict(), os.path.join(save_model_dir, "encoder_claim.bin"))
                torch.save(encoder_evidence.state_dict(), os.path.join(save_model_dir, "encoder_evidence.bin"))
                print("==================================================")
                print("Save Model - Iteration %d" % step)
                print("Save Model - The best f_score is: %.3f" % f_score)
                print("==================================================")

    # finish one epoch, make summary
    avg_train_loss = total_train_loss / len(dataloader_train)
    training_time = utils.format_time(time.time() - t0)
    print("Finish Epoch - Epoch %d" % epoch_i)
    print("Finish Epoch - The best f_score is: %.3f" % best_f_score)
    print("Finish Epoch - The best average loss is: %.3f" % avg_train_loss)
    print("Finish Epoch - The training time is: {:}".format(training_time))

def validate_model(encoder_claim, encoder_evidence, dataloader_validate, dataloader_evidence, topk):
    """
    Validate the model
    """
    # set the model
    encoder_claim.eval()
    encoder_evidence.eval()

    f_scores = []

    for batch in dataloader_validate:
        # move data to cuda
        utils.move_data_to_cuda(batch)

        # predicte the validating data
        prediction = predict_model(encoder_claim, encoder_evidence, 
                                   dataloader_validate, dataloader_evidence,
                                   topk)
        
        # compare the prediction with the real result
        for idx_claim in len(batch["claim_ids"]):
            predict_result = prediction[idx_claim]
            real_result = batch["claim_evidences_ids"][idx_claim]
            # calculate the f-score
            correct_prediction = len(set(predict_result) & set(real_result))
            precision = float(correct_prediction) / len(predict_result)
            recall = float(correct_prediction) / len(real_result)
            f_score = utils.get_fscore(precision, recall)
            f_scores.append(f_score)

    # set the model
    encoder_claim.train()
    encoder_evidence.train()

    # calculate the mean score and return
    return np.mean(f_scores)


def predict_model(encoder_claim, encoder_evidence, dataloader_claims, dataloader_evidence, topk):
    """
    According to the claim text, get the topk evidences.

    :return: evidences (list)
    """
    # predicted evidences
    prediction_result = []

    # encode all the evidences for prediction
    evidence_text_embeddings = []
    evidence_ids = []
    for batch in dataloader_evidence:
        # move data to cuda
        utils.move_data_to_cuda(batch)
        # encode the evidence text(passage)
        curr_evidence_text_embedding = encoder_evidence(input_ids=batch["evidences_input_ids"], 
                                                    attention_mask=batch["evidences_attention_mask"]).last_hidden_state
        # normalize the embeddings
        curr_evidence_text_embedding = nn.functional.normalize(curr_evidence_text_embedding[:, 0, :].detach())
        evidence_text_embeddings.append(curr_evidence_text_embedding)
        evidence_ids.extend(batch["evidences_ids"])
    # combine all evidences' embeddings
    evidence_text_embeddings = torch.cat(evidence_text_embeddings, dim=0)

    # predict all claim text
    for batch in dataloader_claims:
        # move data to cuda
        utils.move_data_to_cuda(batch)
        # encode the claim text
        claim_text_embeddings = encoder_claim(input_ids=batch["claims_texts_input_ids"], 
                                              attention_mask=batch["claims_texts_attention_mask"]).last_hidden_state
        # normalize the embeddings
        claim_text_embeddings = nn.functional.normalize(claim_text_embeddings[:, 0, :])
        # calculate the similarities between claim and evidence
        sims = torch.mm(claim_text_embeddings, evidence_text_embeddings.t())
        # pick the topk similarities' indices
        topk_ids = torch.topk(sims, k=topk, dim=1).indices.tolist()

        # store the topk evidence in current batch
        curr_batch_prediction_result = []
        for idx_claim in len(batch["claim_ids"]):
            for idx_evidence in topk_ids[idx_claim]:
                curr_batch_prediction_result.append(evidence_ids[idx_evidence])
        
        prediction_result.append(curr_batch_prediction_result)
    
    del curr_evidence_text_embedding, evidence_text_embeddings, claim_text_embeddings

    return prediction_result

if __name__ == "main":
    driver_train()