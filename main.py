import numpy as np
from scipy.io import loadmat
import core
import performance


datapath = 'C:\\Users\\fabio\\Desktop\\ING INFORMATICA\\TESI\\DREAMER\\DREAMER_base\\dreamer_base.mat'

data = loadmat(datapath)
EEG_filtt = data['my_base']


# Frequenza di campionamento con cui sono stati presi i dati raw e finestra di campionamento fissata
fs = 128
tw = 10


n_sbjs, n_clips, n_channels, n_samples = EEG_filtt.shape

#n_sbjs = 4


# Creazione di un array per memorizzare la gallery
gallery_data = np.zeros((n_sbjs, 1, n_channels, n_samples))

# Creazione di un array per memorizzare il probing
probing_data = np.zeros((n_sbjs, n_clips - 1, n_channels, n_samples))

print('probing data: ', probing_data.shape)

np.random.seed(1)

# Scelta random di una clip per la gallery per ogni soggetto
for subject in range(n_sbjs):
    gallery_clip_index = np.random.randint(0, n_clips)
    #gallery_clip_index = 0
    #print(gallery_clip_index)

    # Copia dei dati della gallery
    gallery_data[subject] = EEG_filtt[subject, gallery_clip_index, :, :]

    # Creazione del probing senza la clip selezionata per la gallery
    probing_data[subject] = np.delete(EEG_filtt[subject], gallery_clip_index, axis=0)

print("Dimensioni della gallery:", gallery_data.shape)
print("Dimensioni del probing:", probing_data.shape)



gallery_data_3d = np.squeeze(gallery_data, axis=1)
print('Dimensioni del Gallery ridotto: ', gallery_data_3d.shape)



selected_channels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]



features_gallery, v_identity_gallery, n_epochs_gallery = core.compute_features_exp_off(gallery_data_3d, tw, fs, selected_channels)
print("Dimensioni matrice delle features Gallery: ", features_gallery.shape)



EER, AUC = core.compute_EER_AUC(features_gallery, probing_data, tw, fs, selected_channels)

print("All channels -> EER: ",EER, "; AUC: ",AUC)


