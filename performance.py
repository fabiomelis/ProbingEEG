import numpy as np



def reorg(data):

    r_data = np.zeros((data.shape[0] * data.shape[1], data.shape[2]))
    k = 0

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            r_data[k, :] = data[i, j, :]
            k += 1

    return r_data



def calcolo_score(ps, vettore_identita, nEp, nSogg):

    nSeq = 1
    nAcquisizioni = nEp * nSogg
    nScore = nAcquisizioni * (nAcquisizioni - 1) // 2
    score_distanza = np.zeros(nScore)
    flag = np.zeros(nScore)

    print(f'Processing {nAcquisizioni} acquisitions')

    for ind_riga in range(1, nAcquisizioni + 1):  # scorre righe della matrice

        for ind_riga2 in range(ind_riga + 1, nAcquisizioni + 1):

            distanza = np.linalg.norm(ps[ind_riga - 1, :] - ps[ind_riga2 - 1, :])
            score_distanza[nSeq - 1] = 1 / (1 + distanza)
            if vettore_identita[ind_riga - 1] == vettore_identita[ind_riga2 - 1]:  # calcolo del FLAG
                flag[nSeq - 1] = 1  # GENUINO
            else:
                flag[nSeq - 1] = 0  # IMPOSTORE
            nSeq += 1

    return score_distanza, flag


def calcolo_score_gallery(gallery, probing):

    n_seq = 1
    n_acq_gallery, n_features_gallery = gallery.shape
    n_acq_probing, n_features_probing = probing.shape
    n_score = n_acq_probing * (n_acq_probing - 1) // 2
    score_distanza = np.zeros(n_score)
    flag = np.ones(n_score)

    print(f'Processing {n_acq_probing} acquisitions')

    for ind_riga in range(1, n_acq_gallery + 1):  # scorre righe della matrice

        for ind_riga2 in range(1, n_acq_probing + 1):

            distanza = np.linalg.norm(gallery[ind_riga - 1, :] - probing[ind_riga2 - 1, :])
            score_distanza[n_seq - 1] = 1 / (1 + distanza)
            n_seq += 1

    return score_distanza, flag


def calcolo_score_probing(gallery,v_id_gallery, probing, v_id_probing):

    n_seq = 1
    n_acq_gallery, n_features_gallery = gallery.shape
    n_acq_probing, n_features_probing = probing.shape


    nScore = n_acq_probing * (n_acq_probing - 1) // 2
    score_distanza = np.zeros(nScore)
    flag = np.zeros(nScore)

    print(f'Processing {n_acq_probing} acquisitions')

    for ind_riga in range(1, n_acq_gallery + 1):  # scorre righe della matrice

        for ind_riga2 in range(1, n_acq_probing + 1):

            distanza = np.linalg.norm(gallery[ind_riga - 1, :] - probing[ind_riga2 - 1, :])
            score_distanza[n_seq - 1] = 1 / (1 + distanza)

            if v_id_gallery[ind_riga - 1] == v_id_probing[ind_riga2 - 1]:  # calcolo del FLAG
                flag[n_seq - 1] = 1  # GENUINO
            else:
                flag[n_seq - 1] = 0  # IMPOSTORE
            n_seq += 1

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


def calcolo_FRR_gallery(score, flag):
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

        FRR[idx_threshold - 1] = num_gen_rejected / num_genuini
        idx_threshold += 1

    return FRR


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


