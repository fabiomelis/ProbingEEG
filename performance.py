import numpy as np
import core



def calcolo_score_probing(gallery_data_3d, probing, tw, fs, selected_channels,string):

    # probing = (23, 17, 14, 7680)

    if string == 'off':
        features_gallery, v_identity_gallery, n_epochs_gallery = core.compute_features_FOOOF(gallery_data_3d, tw, fs, selected_channels,2,string)

    elif string == 'welch_2':
        features_gallery, v_identity_gallery, n_epochs_gallery = core.compute_features_welch(gallery_data_3d, tw, fs,selected_channels,2,[(0,10),(10,20)])

    print("Dimensioni matrice delle features Gallery: ", features_gallery.shape)

    n_sbjs_gallery, n_epochs_gallery, n_features_gallery = features_gallery.shape
    n_sbjs_probing, n_clips, n_channels, n_samples = probing.shape

    #n_clips = 2

    n_epochs = int(np.floor(n_samples / fs / tw))

    n_seq = 1

    n_score = n_sbjs_probing * n_epochs_gallery * n_sbjs_probing * n_clips
    print('n score: ', n_score)
    score_distanza = np.zeros(n_score)
    flag = np.zeros(n_score)



    for clip in range(n_clips):

        print('Clip analizzata: ', clip + 1,'\n')

        if string == 'off':
            features_probing, v_identity_probing, n_epochs_probing = core.compute_features_FOOOF(probing[:, clip, :, :], tw, fs,
                                                                                             selected_channels, 2, string)
        elif string == 'welch_2':
            features_probing, v_identity_probing, n_epochs_probing = core.compute_features_welch(probing[:, clip, :, :],
                                                                                                 tw, fs,
                                                                                                 selected_channels, 2,
                                                                                                 [(0,10),(10,20)])

        print('features_probing: ', features_probing.shape)



        for ind_sbj_probing in range(1, n_sbjs_probing + 1):


            for ind_epoch_probing in range(1, n_epochs_probing + 1):

                print(f'    Trying probing sbj {ind_sbj_probing}, epoch {ind_epoch_probing}','\n')


                for ind_sbj_gallery in range(1, n_sbjs_gallery + 1):

                    print(f'        Gallery sbj {ind_sbj_gallery}, {n_epochs_gallery} epochs')

                    best_score = 0


                    for ind_epoch_gallery in range(1, n_epochs_gallery + 1):


                        distanza = np.linalg.norm(features_gallery[ind_sbj_gallery - 1, ind_epoch_gallery - 1, :] - features_probing[ind_sbj_probing - 1, ind_epoch_probing - 1, :])

                        score = 1 / (1 + distanza)


                        #print('score: ', score)

                        if score > best_score:
                            best_score = score

                        #print('best score: ', best_score)

                    print(f'        best score saved for sbj{ind_sbj_gallery}: ', best_score)

                    score_distanza[n_seq - 1] = best_score

                    #print('score vector: ', score_distanza)

                    if ind_sbj_gallery == ind_sbj_probing:  # calcolo del FLAG
                        flag[n_seq - 1] = 1  # GENUINO
                        print('         Genuino\n')
                    else:
                        flag[n_seq - 1] = 0  # IMPOSTORE
                        print('         Impostore\n')

                    #print('flag: ', flag)

                    n_seq += 1


    print('n_seq:', n_seq)

    return score_distanza, flag





def calcolo_FAR_FRR(score, flag):
    lunghezza_score = len(score)
    lunghezza_flag = len(flag)
    idx_threshold = 1
    num_genuini = np.sum(flag)
    num_impostori = lunghezza_flag - num_genuini

    vettore_soglia = np.arange(0, 1.01, 0.01)
    FRR = np.zeros_like(vettore_soglia)
    FAR = np.zeros_like(vettore_soglia)

    for soglia in vettore_soglia:
        num_gen_rejected = 0  # genuini rifiutati
        num_imp_accepted = 0  # impostori accettati
        for ind_scoreprova in range(lunghezza_score):
            if score[ind_scoreprova] < soglia and flag[ind_scoreprova] == 1:
                num_gen_rejected += 1  # numero dei genuini rifiutati
            if score[ind_scoreprova] > soglia and flag[ind_scoreprova] == 0:
                num_imp_accepted += 1  # numero degli impostori accettati

        FRR[idx_threshold - 1] = num_gen_rejected / num_genuini
        FAR[idx_threshold - 1] = num_imp_accepted / num_impostori
        idx_threshold += 1

    return FAR, FRR, vettore_soglia




def calcolo_EER_AUC(FAR, FRR):

    # Calcolo di AUC
    AUC = np.abs(np.trapz(FAR, FRR))

    # Calcolo di EER
    valore_minimo = np.abs(FAR - FRR)
    indice_minimo = np.argmin(valore_minimo)

    EER_FAR = FAR[indice_minimo]
    EER_FRR = FRR[indice_minimo]

    EER = (EER_FAR + EER_FRR) / 2

    return EER, AUC


