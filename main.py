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


# Creazione di un array per memorizzare la gallery
gallery_data = np.zeros((n_sbjs, 1, n_channels, n_samples))

# Creazione di un array per memorizzare il probing
probing_data = np.zeros((n_sbjs, n_clips - 1, n_channels, n_samples))


# Scelta random di una clip per la gallery per ogni soggetto
for subject in range(n_sbjs):
    gallery_clip_index = np.random.randint(0, n_clips)

    # Copia dei dati della gallery
    gallery_data[subject] = EEG_filtt[subject, gallery_clip_index, :, :]

    # Creazione del probing senza la clip selezionata per la gallery
    probing_data[subject] = np.delete(EEG_filtt[subject], gallery_clip_index, axis=0)

print("Dimensioni della gallery:", gallery_data.shape)
print("Dimensioni del probing:", probing_data.shape)


gallery_data_3d = np.squeeze(gallery_data, axis=1)
print('Dimensioni del Gallery ridotto: ', gallery_data_3d.shape)


# Concatenamento delle clip del Probing
probing_data_concatenate = np.concatenate(np.split(probing_data, n_clips - 1, axis=1), axis=3)

print('Dimensioni del Probing concatenato: ',probing_data_concatenate.shape)

probing_data_3d = probing_data_concatenate[:, 0, :, :]

print('Dimensioni del Probing ridotto: ', probing_data_3d.shape)


selected_channels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]


# Ottenimento delle matrici delle features per Gallery e Probing

features_gallery, v_identity_gallery, n_epochs_gallery = core.compute_features_exp_off(gallery_data_3d, tw, fs, selected_channels)
print("Dimensioni matrice delle features Gallery: ", features_gallery.shape)

features_probing, v_identity_probing, n_epochs_probing = core.compute_features_exp_off(probing_data_3d, tw, fs, selected_channels)
print("Dimensioni matrice delle features Probing: ", features_probing.shape)


'''
# Calcolo FRR per ogni soggetto (?)

for subject in range(n_sbjs):

    # Estrazione delle caratteristiche di gallery e probing per il soggetto corrente
    gallery_subject_features = features_gallery[subject]
    probing_subject_features = features_probing[subject]

    print('Dimensioni in ingresso a calcolo score per Probing: ', probing_subject_features.shape)

    score, flag = performance.calcolo_score_gallery(gallery_subject_features,probing_subject_features)

    FRR = performance.calcolo_FRR_gallery(score, flag)

    print("Soggetto ", subject, " -> False Reject Rate: ",FRR)
'''


# Trasformazione matrice 3d delle features in matrice 2d con acquisizioni (sogg * epoche) e features
acq_gallery = performance.reorg(features_gallery)
acq_probing = performance.reorg(features_probing)


EER, AUC = core.compute_EER_AUC(acq_gallery, v_identity_gallery, acq_probing, v_identity_probing)
print("All channels -> EER: ",EER, "; AUC: ",AUC)



selected_channels = [1, 4, 6, 9, 12]   # Migliori canali selezionati comparando le 18 clips

features_gallery, v_identity_gallery, n_epochs_gallery = core.compute_features_exp_off(gallery_data_3d, tw, fs, selected_channels)

features_probing, v_identity_probing, n_epochs_probing = core.compute_features_exp_off(probing_data_3d, tw, fs, selected_channels)


acq_gallery = performance.reorg(features_gallery)
acq_probing = performance.reorg(features_probing)


EER, AUC = core.compute_EER_AUC(acq_gallery, v_identity_gallery, acq_probing, v_identity_probing)
print("Best 5 channels -> EER: ",EER, "; AUC: ",AUC)