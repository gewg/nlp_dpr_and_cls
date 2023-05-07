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
from tqdm import tqdm
import utils
import time
import datetime
import os
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from transformers import get_linear_schedule_with_warmup

debug = False
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

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
    train_dataset = utils.TrainDataset(params.num_evidence_per_batch, tokenizer, "data/train-claims.json", "data/evidence.json")
    evidence_dataset = utils.EvidenceDataset(tokenizer, "data/evidence.json")
    validate_dataset = utils.ValidateDataset(tokenizer, "data/dev-claims.json")
    # train_dataset = utils.TrainDataset(params.num_evidence_per_batch, tokenizer, "data/train-claims-debug.json", "data/evidence-debug.json")
    # evidence_dataset = utils.EvidenceDataset(tokenizer, "data/evidence-debug.json")
    # validate_dataset = utils.ValidateDataset(tokenizer, "data/dev-claims.json")

    # initialize the dataloaderß
    dataloader_train = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=False, collate_fn=train_dataset.collate_fn)
    dataloader_evidence = DataLoader(evidence_dataset, batch_size=params.batch_size_evidences, shuffle=False, collate_fn=evidence_dataset.collate_fn)
    dataloader_validate = DataLoader(validate_dataset, batch_size=params.batch_size, shuffle=False, collate_fn=validate_dataset.collate_fn)

    # create the encoders, there are two encoders for 'question' and 'passage' according to dense passage retrieval
    encoder_claim = AutoModel.from_pretrained(params.tokenizer)
    encoder_evidence = AutoModel.from_pretrained(params.tokenizer)
    encoder_claim.cuda()
    encoder_evidence.cuda()

    # create the optimizer
    optimizer_claim = optim.Adam(encoder_claim.parameters())
    optimizer_evidence = optim.Adam(encoder_evidence.parameters())

    # create the learning rate scheduler
    total_steps = len(dataloader_train) * params.num_epoch
    scheduler_claim = get_linear_schedule_with_warmup(optimizer=optimizer_claim, num_warmup_steps=params.num_warmup_steps, num_training_steps=total_steps)
    scheduler_evidence = get_linear_schedule_with_warmup(optimizer=optimizer_evidence, num_warmup_steps=params.num_warmup_steps, num_training_steps=total_steps)

    '''Start training'''
    for epoch_i in range(0, params.num_epoch):

        print("")
        print("======== Epoch %d / %d ========" % (epoch_i+1, params.num_epoch))

        train_per_epoch(dataloader_train, dataloader_evidence, dataloader_validate,
          encoder_claim, encoder_evidence,
          optimizer_claim, optimizer_evidence,
          scheduler_claim, scheduler_evidence,
          params.similarity_adjustment,
          params.grad_norm,
          params.num_itr_validate,
          params.num_topk,
          params.save_model_dir,
          epoch_i+1)
        
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
        print("Train - Start training")

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
        sims = torch.mm(claim_text_embeddings, evidence_text_embeddings.t())
        # calculate the loss for each evidence
        losses = - nn.functional.log_softmax(sims / similarity_adjustment, dim=1)  # while the difference between similarities may be small, the similarity_adjustment is used to increase it

        # for each claim text, get the positive evidence's loss
        each_claim_loss = []
        for idx, positive_evidences_positions in enumerate(batch["claims_positive_evidences_positions"]):
            curr_claim_pos_evidence_start = positive_evidences_positions[0]
            curr_claim_pos_evidence_end = positive_evidences_positions[1]
            curr_claim_losses = losses[idx, curr_claim_pos_evidence_start:curr_claim_pos_evidence_end+1]
            # while each claim has more than one positive evidence, so calculate the mean as the loss
            each_claim_loss.append(torch.mean(curr_claim_losses))

        # reset the gradient
        optimizer_claim.zero_grad()
        optimizer_evidence.zero_grad()

        # backward the loss
        each_claim_loss = torch.mean(torch.stack(each_claim_loss))
        each_claim_loss.backward()

        # count the total loss
        total_train_loss += each_claim_loss.item()

        # gradient clipping
        nn.utils.clip_grad_norm_(encoder_claim.parameters(), grad_norm)
        nn.utils.clip_grad_norm_(encoder_evidence.parameters(), grad_norm)

        # update the parameters
        optimizer_claim.step()
        optimizer_evidence.step()

        # update the learning rate
        scheduler_claim.step()
        scheduler_evidence.step()

        del claim_text_embeddings, evidence_text_embeddings, sims, losses

        print("Train - Finish training")
        print("")

        # validate and report the information per 'num_itr_validate' times iteration
        if step % num_itr_validate == 0 and not step == 0:
            
            # validate the model
            print("==================================================")
            print("Validate - Start validating")

            f_score = validate_model(encoder_claim, encoder_evidence, 
                                     dataloader_validate, dataloader_evidence, 
                                     topk)

            print("Validate - Iteration %d" % step)
            print("Validate - The f_score is: %.3f" % f_score)
            print("Validate - Current best f_score is: %.3f" % best_f_score)
            print("==================================================")
            print("")

            # if the f-score is better, save the model
            if f_score > best_f_score:
                best_f_score = f_score
                torch.save(encoder_claim.state_dict(), os.path.join(save_model_dir, "%.3f_encoder_claim.bin" % f_score))
                torch.save(encoder_evidence.state_dict(), os.path.join(save_model_dir, "%.3f_encoder_evidence.bin" % f_score))
                print("==================================================")
                print("Save Model - Iteration %d" % step)
                print("Save Model - The best f_score is: %.3f" % f_score)
                print("==================================================")

    # finish one epoch, make summary
    avg_train_loss = total_train_loss / len(dataloader_train)
    training_time = utils.format_time(time.time() - t0)
    print("Finish Epoch - Epoch %d" % epoch_i)
    print("Finish Epoch - The best f_score is: %.3f" % best_f_score)
    print("Finish Epoch - The average loss is: %.3f" % avg_train_loss)
    print("Finish Epoch - The training time is: {:}".format(training_time))

def validate_model(encoder_claim, encoder_evidence, dataloader_validate, dataloader_evidence, topk):
    """
    Validate the model
    """
    # set the model
    # encoder_claim.eval()
    # encoder_evidence.eval()

    # encode the evidence
    if not debug:
        evidence_ids, evidence_text_embeddings = encode_evidences(encoder_evidence, dataloader_evidence)
        evidence_embedding_dict = {"evidence_ids": evidence_ids, "evidence_text_embeddings": evidence_text_embeddings}
        np.save("data/evidence_embedding.npy", evidence_embedding_dict)  # save embeddings to file for debug
    # if debug, directly get evidences' embeddings from file
    else:
        print("Debug - Loading evidence_embedding")
        evidence_embedding_dict = np.load("data/evidence_embedding.npy").item()
        evidence_ids = evidence_embedding_dict["evidence_ids"]
        evidence_text_embeddings = evidence_embedding_dict["evidence_text_embeddings"]

    f_scores = []
    
    # predicte the validating data
    prediction = predict_model(encoder_claim, dataloader_validate,
                                evidence_ids, evidence_text_embeddings,
                                topk)

    # compare the prediction with the real result
    for batch in tqdm(dataloader_validate):
        # move data to cuda
        utils.move_data_to_cuda(batch)
        
        for idx, claim_id in enumerate(batch["claims_ids"]):
            # calculate the f-score
            predict_result = prediction[claim_id]
            real_result = batch["claims_evidences_ids"][idx]
            correct_prediction = len(set(predict_result) & set(real_result))
            # avoid denominator is 0
            if correct_prediction != 0:
                precision = float(correct_prediction) / len(predict_result)
                recall = float(correct_prediction) / len(real_result)
                f_score = utils.get_fscore(precision, recall)
                f_scores.append(f_score)

    del evidence_text_embeddings

    # set the model
    # encoder_claim.train()
    # encoder_evidence.train()

    # set a warning
    if not f_scores:
        print("Validate - !!! No correct prediction for this turn !!!")
        f_scores = [0]

    # calculate the mean score and return
    return np.mean(f_scores)


def encode_evidences(encoder_evidence, dataloader_evidence):
    """
    Encode all the evidences. Because the dataset is too large, so the encoding is extracted
    """

    # encode all the evidences for prediction
    print("Predict - Start encoding evidences")
    evidence_text_embeddings = []
    evidence_ids = []
    for batch in tqdm(dataloader_evidence):
        # move data to cuda
        utils.move_data_to_cuda(batch)
        # encode the evidence text(passage)
        curr_evidence_text_embedding = encoder_evidence(input_ids=batch["evidences_input_ids"], 
                                                        attention_mask=batch["evidences_attention_mask"]).last_hidden_state
        print(curr_evidence_text_embedding)
        # normalize the embeddings
        curr_evidence_text_embedding = nn.functional.normalize(curr_evidence_text_embedding[:, 0, :].detach())
        evidence_text_embeddings.append(curr_evidence_text_embedding)
        evidence_ids.extend(batch["evidences_ids"])

    # combine all evidences' embeddings
    evidence_text_embeddings = torch.cat(evidence_text_embeddings, dim=0)

    del curr_evidence_text_embedding

    return [evidence_ids, evidence_text_embeddings]



def predict_model(encoder_claim, dataloader_claims, evidence_ids, evidence_text_embeddings, topk):
    """
    According to the claim text, get the topk evidences.

    :return: evidences (list)
    """
    # predicted evidences
    prediction_result = defaultdict(lambda: [])

    # predict all claim text
    print("Predict - Start predicting claims' evidences")
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
        for idx, claim_id in enumerate(batch["claims_ids"]):
            for idx_evidence in topk_ids[idx]:
                prediction_result[claim_id].append(evidence_ids[idx_evidence])
                
    print(sims)
    del claim_text_embeddings, sims

    return prediction_result

if __name__ == "__main__":
    print("Starting...")
    driver_train()