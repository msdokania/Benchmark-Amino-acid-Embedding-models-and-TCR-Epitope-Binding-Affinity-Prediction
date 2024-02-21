from TCRpeg import TCRpeg
import pandas as pd


dat = pd.read_csv('/home/monalisadokania/TCRpeg/TCREpitopePairs.csv')
dat['tcr_embeds'] = None
dat['epi_embeds'] = None

tcrs = dat['tcr'].tolist()
epis = dat['epi'].tolist()

model_tcr = TCRpeg(embedding_path='data/embedding_32.txt',load_data=True, path_train=tcrs)
model_epi = TCRpeg(embedding_path='data/embedding_32.txt',load_data=True, path_train=epis)


model_tcr.create_model() #initialize the TCRpeg model
model_epi.create_model()

model_tcr.train_tcrpeg(epochs=20, batch_size= 30, lr=1e-3)
model_epi.train_tcrpeg(epochs=20, batch_size= 30, lr=1e-3)


dat['tcr_embeds'] = model_tcr.get_embedding(tcrs).tolist()
dat['epi_embeds'] = model_epi.get_embedding(epis).tolist()

dat.to_pickle("/home/monalisadokania/TCRpeg/embeddings/tcrpeg_embed.pkl")
