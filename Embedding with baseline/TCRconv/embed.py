from bert_mdl import retrieve_model
import torch
import pandas as pd
import numpy as np


model = retrieve_model()
model.cuda()
model.to(torch.float32)
dat = pd.read_csv('./TCREpitopePairs.csv')

tcr = dat['tcr'].tolist()
epi = dat['epi'].tolist()
dat['tcr_embeds'] = None
dat['epi_embeds'] = None
tcr_embs, epi_embs = [], []

inputs_tcr = model.tokenizer.batch_encode_plus(tcr, add_special_tokens=False, padding=True, truncation=True, max_length=model.hparams.max_length)
inputs_epi = model.tokenizer.batch_encode_plus(epi, add_special_tokens=False, padding=True, truncation=True, max_length=model.hparams.max_length)

model.extract_emb = True
model.eval()
assert len(tcr) == len(epi)
for i in range(int(np.ceil(len(tcr)/ 100))):
    cur_inputs_tcr = {}
    cur_inputs_epi = {}

    cur_tcr = tcr[i * 100:(i + 1) * 100]
    cur_epi = epi[i * 100:(i + 1) * 100]
    
    cur_inputs_tcr['input_ids'] = inputs_tcr['input_ids'][i * 100:(i + 1) * 100]
    cur_inputs_epi['input_ids'] = inputs_epi['input_ids'][i * 100:(i + 1) * 100]

    cur_inputs_tcr['token_type_ids'] = inputs_tcr['token_type_ids'][i * 100:(i + 1) * 100]
    cur_inputs_epi['token_type_ids'] = inputs_epi['token_type_ids'][i * 100:(i + 1) * 100]

    cur_inputs_tcr['attention_mask'] = inputs_tcr['attention_mask'][i * 100:(i + 1) * 100]
    cur_inputs_epi['attention_mask'] = inputs_epi['attention_mask'][i * 100:(i + 1) * 100]

    
    embedding_tcr = model.forward(**cur_inputs_tcr)
    embedding_epi = model.forward(**cur_inputs_epi)

    embedding_tcr = embedding_tcr.cpu().numpy().tolist()
    embedding_epi = embedding_epi.cpu().numpy().tolist()

    tcr_embs.extend(embedding_tcr)
    epi_embs.extend(embedding_epi)

print(len(tcr_embs))
dat['tcr_embeds'] = [pd.Series(row) for row in tcr_embs]
dat['epi_embeds'] = [pd.Series(row) for row in epi_embs]

    

dat.to_pickle("./embeddings/tcrconv_embed.pkl")




