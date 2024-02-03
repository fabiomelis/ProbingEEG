import numpy as np
from scipy.signal import welch
from fooof import FOOOF


def comp_aperiodic_and_first_peak(tmp_data, i_sbj, i_channel, n_ch_sel, fs, n_features):

    # Inizializza il modello FOOOF con il parametro peak_width_limits modificato
    fooof_model = FOOOF(peak_width_limits=[1.5, 8])

    indices = []
    for i in range(1, n_features + 1):
        index = (n_ch_sel * n_features) - n_features + i - 1
        indices.append(index)

    index1, index2, index3, index4, index5 = indices

    # Ottieni il vettore degli n_steps estraendolo dalla matrice
    tmp_signal = np.squeeze(tmp_data[i_sbj - 1, i_channel - 1, :])

    # Per evitare errori dovuti alla frequenza bisogna impostare nperseg a 255 e non a 256 come di default
    frequencies, power_spectrum = welch(tmp_signal, fs=fs, nperseg=min(255, len(tmp_signal)))

    # Applica il modello FOOOF
    fooof_model.fit(frequencies, power_spectrum, freq_range=[2, 60])

    aperiodic_params = fooof_model.aperiodic_params_
    peak_params = fooof_model.peak_params_

    # Estraggo valori del fit aperiodico
    exp = aperiodic_params[0]
    offset = aperiodic_params[1]

    # Estraggo valori di frequenza, potenza e larghezza del primo picco
    freq_1peak = peak_params[0, 0]
    power_1peak = peak_params[0, 1]
    width_1peak = peak_params[0, 2]

    return exp, offset, freq_1peak, power_1peak, width_1peak, [index1, index2, index3, index4, index5]


def comp_first_peak(tmp_data, i_sbj, i_channel, n_ch_sel, fs, n_features):

    # Inizializza il modello FOOOF con il parametro peak_width_limits modificato
    fooof_model = FOOOF(peak_width_limits=[1.5, 8])

    indices = []
    for i in range(1, n_features + 1):
        index = (n_ch_sel * n_features) - n_features + i - 1
        indices.append(index)

    index1, index2, index3 = indices

    # Ottieni il vettore degli n_steps estraendolo dalla matrice
    tmp_signal = np.squeeze(tmp_data[i_sbj - 1, i_channel - 1, :])

    # Per evitare errori dovuti alla frequenza bisogna impostare nperseg a 255 e non a 256 come di default
    frequencies, power_spectrum = welch(tmp_signal, fs=fs, nperseg=min(255, len(tmp_signal)))

    # Applica il modello FOOOF
    fooof_model.fit(frequencies, power_spectrum, freq_range=[2, 60])

    peak_params = fooof_model.peak_params_

    # Estraggo valori di frequenza, potenza e larghezza del primo picco
    freq_1peak = peak_params[0, 0]
    power_1peak = peak_params[0, 1]
    width_1peak = peak_params[0, 2]

    return freq_1peak, power_1peak, width_1peak, [index1, index2, index3]


def comp_first_peak_freq(tmp_data, i_sbj, i_channel, n_ch_sel, fs, n_features):

    # Inizializza il modello FOOOF con il parametro peak_width_limits modificato
    fooof_model = FOOOF(peak_width_limits=[1.5, 8], aperiodic_mode='knee')

    # Calcola gli indici dinamicamente in base a n_features
    indices = []
    for i in range(1, n_features + 1):
        index = (n_ch_sel * n_features) - n_features + i - 1
        indices.append(index)

    index1 = indices

    # Ottieni il vettore degli n_steps estraendolo dalla matrice
    tmp_signal = np.squeeze(tmp_data[i_sbj - 1, i_channel - 1, :])

    # Per evitare errori dovuti alla frequenza bisogna impostare nperseg a 255 e non a 256 come di default
    frequencies, power_spectrum = welch(tmp_signal, fs=fs, nperseg=min(255, len(tmp_signal)))

    # Applica il modello FOOOF
    fooof_model.fit(frequencies, power_spectrum, freq_range=[2, 60])

    peak_params = fooof_model.peak_params_

    # Estraggo valori di frequenza, potenza e larghezza del primo picco
    freq_1peak = peak_params[0, 0]


    return freq_1peak, index1

def comp_first_peak_pw(tmp_data, i_sbj, i_channel, n_ch_sel, fs, n_features):

    # Inizializza il modello FOOOF con il parametro peak_width_limits modificato
    fooof_model = FOOOF(peak_width_limits=[1.5, 8])

    # Calcola gli indici dinamicamente in base a n_features
    indices = []
    for i in range(1, n_features + 1):
        index = (n_ch_sel * n_features) - n_features + i - 1
        indices.append(index)

    index1 = indices

    # Ottieni il vettore degli n_steps estraendolo dalla matrice
    tmp_signal = np.squeeze(tmp_data[i_sbj - 1, i_channel - 1, :])

    # Per evitare errori dovuti alla frequenza bisogna impostare nperseg a 255 e non a 256 come di default
    frequencies, power_spectrum = welch(tmp_signal, fs=fs, nperseg=min(255, len(tmp_signal)))

    # Applica il modello FOOOF
    fooof_model.fit(frequencies, power_spectrum, freq_range=[2, 60])

    peak_params = fooof_model.peak_params_

    # Estraggo valori di frequenza, potenza e larghezza del primo picco
    pw_1peak = peak_params[0, 2]


    return pw_1peak, index1


def comp_aperiodic_1st_2nd_freq(tmp_data, i_sbj, i_channel, n_ch_sel, fs, n_features):

    # Inizializza il modello FOOOF con il parametro peak_width_limits modificato
    fooof_model = FOOOF()

    indices = []
    for i in range(1, n_features + 1):
        index = (n_ch_sel * n_features) - n_features + i - 1
        indices.append(index)

    index1, index2, index3, index4 = indices

    # Ottieni il vettore degli n_steps estraendolo dalla matrice
    tmp_signal = np.squeeze(tmp_data[i_sbj - 1, i_channel - 1, :])

    # Per evitare errori dovuti alla frequenza bisogna impostare nperseg a 255 e non a 256 come di default
    frequencies, power_spectrum = welch(tmp_signal, fs=fs, nperseg=min(255, len(tmp_signal)))

    # Applica il modello FOOOF
    fooof_model.fit(frequencies, power_spectrum, freq_range=[2, 60])

    aperiodic_params = fooof_model.aperiodic_params_
    peak_params = fooof_model.peak_params_

    # Estraggo valori del fit aperiodico
    exp = aperiodic_params[0]
    offset = aperiodic_params[1]

    # Estraggo valori di frequenza, potenza e larghezza del primo picco
    freq_1peak = peak_params[0, 0]
    freq_2peak = peak_params[1, 0]

    return exp, offset, freq_1peak, freq_2peak, [index1, index2, index3, index4]

def comp_peak_1st_2nd_freq(tmp_data, i_sbj, i_channel, n_ch_sel, fs, n_features):

    # Inizializza il modello FOOOF con il parametro peak_width_limits modificato
    fooof_model = FOOOF(peak_width_limits=[1, 8])

    indices = []
    for i in range(1, n_features + 1):
        index = (n_ch_sel * n_features) - n_features + i - 1
        indices.append(index)

    index1, index2, = indices

    # Ottieni il vettore degli n_steps estraendolo dalla matrice
    tmp_signal = np.squeeze(tmp_data[i_sbj - 1, i_channel - 1, :])

    # Per evitare errori dovuti alla frequenza bisogna impostare nperseg a 255 e non a 256 come di default
    frequencies, power_spectrum = welch(tmp_signal, fs=fs, nperseg=min(255, len(tmp_signal)))

    # Applica il modello FOOOF
    fooof_model.fit(frequencies, power_spectrum, freq_range=[2, 60])

    #aperiodic_params = fooof_model.aperiodic_params_
    peak_params = fooof_model.peak_params_


    # Estraggo valori di frequenza, potenza e larghezza del primo picco
    freq_1peak = peak_params[0, 0]
    freq_2peak = peak_params[1, 0]

    return freq_1peak, freq_2peak, [index1, index2]

def comp_peak_1st_2nd_pw(tmp_data, i_sbj, i_channel, n_ch_sel, fs, n_features):

    # Inizializza il modello FOOOF con il parametro peak_width_limits modificato
    fooof_model = FOOOF(peak_width_limits=[1, 8])

    indices = []
    for i in range(1, n_features + 1):
        index = (n_ch_sel * n_features) - n_features + i - 1
        indices.append(index)

    index1, index2, = indices

    # Ottieni il vettore degli n_steps estraendolo dalla matrice
    tmp_signal = np.squeeze(tmp_data[i_sbj - 1, i_channel - 1, :])

    # Per evitare errori dovuti alla frequenza bisogna impostare nperseg a 255 e non a 256 come di default
    frequencies, power_spectrum = welch(tmp_signal, fs=fs, nperseg=min(255, len(tmp_signal)))

    # Applica il modello FOOOF
    fooof_model.fit(frequencies, power_spectrum, freq_range=[2, 60])

    #aperiodic_params = fooof_model.aperiodic_params_
    peak_params = fooof_model.peak_params_


    # Estraggo valori di frequenza, potenza e larghezza del primo picco
    pw_1peak = peak_params[0, 2]
    pw_2peak = peak_params[1, 2]

    return pw_1peak, pw_2peak, [index1, index2]


def comp_aperiodic_and_1st_freq(tmp_data, i_sbj, i_channel, n_ch_sel, fs, n_features):

    # Inizializza il modello FOOOF con il parametro peak_width_limits modificato
    fooof_model = FOOOF(peak_width_limits=[1.5, 8])

    indices = []
    for i in range(1, n_features + 1):
        index = (n_ch_sel * n_features) - n_features + i - 1
        indices.append(index)

    index1, index2, index3 = indices

    # Ottieni il vettore degli n_steps estraendolo dalla matrice
    tmp_signal = np.squeeze(tmp_data[i_sbj - 1, i_channel - 1, :])

    # Per evitare errori dovuti alla frequenza bisogna impostare nperseg a 255 e non a 256 come di default
    frequencies, power_spectrum = welch(tmp_signal, fs=fs, nperseg=min(255, len(tmp_signal)))

    # Applica il modello FOOOF
    fooof_model.fit(frequencies, power_spectrum, freq_range=[2, 60])

    aperiodic_params = fooof_model.aperiodic_params_
    peak_params = fooof_model.peak_params_

    # Estraggo valori del fit aperiodico
    exp = aperiodic_params[0]
    offset = aperiodic_params[1]

    # Estraggo valori di frequenza, potenza e larghezza del primo picco
    freq_1peak = peak_params[0, 0]


    return exp, offset, freq_1peak, [index1, index2, index3]


def comp_aperiodic(tmp_data, i_sbj, i_channel, n_ch_sel, fs, n_features):
    # Inizializza il modello FOOOF con il parametro peak_width_limits modificato
    fooof_model = FOOOF(peak_width_limits=[1.5, 8])

    indices = []
    for i in range(1, n_features + 1):
        index = (n_ch_sel * n_features) - n_features + i - 1
        indices.append(index)

    index1, index2 = indices

    # Ottieni il vettore degli n_steps estraendolo dalla matrice
    tmp_signal = np.squeeze(tmp_data[i_sbj - 1, i_channel - 1, :])

    # Per evitare errori dovuti alla frequenza bisogna impostare nperseg a 255 e non a 256 come di default
    frequencies, power_spectrum = welch(tmp_signal, fs=fs, nperseg=min(255, len(tmp_signal)))

    # Applica il modello FOOOF
    fooof_model.fit(frequencies, power_spectrum, freq_range=[2, 60])

    aperiodic_params = fooof_model.aperiodic_params_
    #peak_params = fooof_model.peak_params_

    # Estraggo valori del fit aperiodico
    exp = aperiodic_params[0]
    offset = aperiodic_params[1]

    return exp, offset, [index1, index2]


def comp_aperiodic_offset(tmp_data, i_sbj, i_channel, n_ch_sel, fs, n_features):

    # Inizializza il modello FOOOF con il parametro peak_width_limits modificato
    fooof_model = FOOOF(peak_width_limits=[1.5, 8])

    indices = []
    for i in range(1, n_features + 1):
        index = (n_ch_sel * n_features) - n_features + i - 1
        indices.append(index)

    index1 = indices

    # Ottieni il vettore degli n_steps estraendolo dalla matrice
    tmp_signal = np.squeeze(tmp_data[i_sbj - 1, i_channel - 1, :])

    # Per evitare errori dovuti alla frequenza bisogna impostare nperseg a 255 e non a 256 come di default
    frequencies, power_spectrum = welch(tmp_signal, fs=fs, nperseg=min(255, len(tmp_signal)))

    # Applica il modello FOOOF
    fooof_model.fit(frequencies, power_spectrum, freq_range=[2, 60])

    aperiodic_params = fooof_model.aperiodic_params_
    #peak_params = fooof_model.peak_params_

    # Estraggo valori del fit aperiodico
    offset = aperiodic_params[1]

    return offset, index1


def comp_aperiodic_exp(tmp_data, i_sbj, i_channel, n_ch_sel, fs, n_features):

    # Inizializza il modello FOOOF con il parametro peak_width_limits modificato
    fooof_model = FOOOF(peak_width_limits=[2, 8])

    indices = []
    for i in range(1, n_features + 1):
        index = (n_ch_sel * n_features) - n_features + i - 1
        indices.append(index)

    index1 = indices

    # Ottieni il vettore degli n_steps estraendolo dalla matrice
    tmp_signal = np.squeeze(tmp_data[i_sbj - 1, i_channel - 1, :])


    # Per evitare errori dovuti alla frequenza bisogna impostare nperseg a 255 e non a 256 come di default
    frequencies, power_spectrum = welch(tmp_signal, fs=fs, nperseg=min(255, len(tmp_signal)))



    # Applica il modello FOOOF
    fooof_model.fit(frequencies, power_spectrum, freq_range=[2, 60])

    aperiodic_params = fooof_model.aperiodic_params_

    # Estraggo valori del fit aperiodico
    exp = aperiodic_params[0]

    #utils.plot_ps_welch(frequencies, power_spectrum)

    # Plotta il modello FOOOF
    #fooof_model.plot()

    # Mostra il grafico
    #plt.show()

    return exp, index1

