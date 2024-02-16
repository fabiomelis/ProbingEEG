import numpy as np
from scipy.io import loadmat
import core
import performance
import selection_alg
import compare


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



#selected_channels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
selected_channels = [6, 1, 2, 5, 14, 8, 13, 3, 9, 11, 10, 7, 12, 4]
#selected_channels = [6, 1, 2, 5, 13, 8]
#selected_channels = [1, 4, 6, 9, 12]





#core.compute_EER_fixed_list(selected_channels,gallery_data_3d,probing_data,tw,fs,'welch_2')
selection_alg.forward_selection_eer(len(selected_channels),gallery_data_3d,probing_data,tw,fs,'welch_2')
#EER, AUC = core.compute_EER_AUC(gallery_data_3d, probing_data, tw, fs, selected_channels,'welch_2')


#print("Best Channels [6, 1, 2, 5, 13, 8] -> EER: ",EER, "; AUC: ",AUC)
#print("All Channels -> EER: ",EER, "; AUC: ",AUC)


