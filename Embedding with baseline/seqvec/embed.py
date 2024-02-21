import pandas as pd
from pathlib import Path
import torch
from allennlp.commands.elmo import ElmoEmbedder

model_dir = Path('/home/monalisadokania/SeqVec')
weights = model_dir/'weights.hdf5'
options = model_dir/'options.json'
embedder  = ElmoEmbedder(options,weights,cuda_device=0) # cuda_device=-1 for CPU
 
def SeqVec_embedding(x):
    list_of_tensors = embedder.embed_sentences(x)
    embed_list = []
    for t in list_of_tensors:
        embed_list.append(torch.tensor(t).sum(dim=0).mean(dim=0).tolist())
    return embed_list 

dat = pd.read_csv('/home/monalisadokania/SeqVec/TCREpitopePairs.csv')
dat['tcr_embeds'] = None
dat['epi_embeds'] = None


character_lists_epi = dat['epi'].apply(list)
list_of_lists_epi = character_lists_epi.tolist()
character_lists_tcr = dat['tcr'].apply(list)
list_of_lists_tcr = character_lists_tcr.tolist()

dat['epi_embeds'] = SeqVec_embedding(list_of_lists_epi)
dat['tcr_embeds'] = SeqVec_embedding(list_of_lists_tcr)

dat.to_pickle("/home/monalisadokania/SeqVec/embeddings/seqvec_embed.pkl")
