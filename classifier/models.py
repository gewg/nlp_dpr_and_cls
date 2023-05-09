from torch import nn

class ClaimClassifier(nn.Module):

    def __init__(self, pretrained_model):
        super(ClaimClassifier, self).__init__()

        self.pretrained_model = pretrained_model
        self.hidden_layer_size = self.pretrained_model.config.hidden_size
        self.layer = nn.Sequential(
            nn.Linear(self.hidden_layer_size, self.hidden_layer_size),
            nn.Tanh(),
            nn.Linear(self.hidden_layer_size, 4)
        )
    
    def forward(self, claims_texts_input_ids, claims_texts_attention_mask):
        # extract features
        claim_text_embeddings = self.pretrained_model(input_ids=claims_texts_input_ids, 
                                              attention_mask=claims_texts_attention_mask).last_hidden_state
        claim_text_embeddings = claim_text_embeddings[:, 0, :]
        # linear transformation
        labels = self.layer(claim_text_embeddings)

        return labels
