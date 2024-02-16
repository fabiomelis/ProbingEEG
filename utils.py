import numpy as np
import matplotlib.pyplot as plt


def plot_ps_welch(frequencies, power_spectrum):
    plt.figure(figsize=(10, 6))
    plt.semilogy(frequencies, power_spectrum)
    plt.title('Spettro di Potenza con Metodo di Welch')
    plt.xlabel('Frequenza (Hz)')
    plt.ylabel('Densità di Potenza')
    plt.grid(True)
    plt.show()


def find_ps_fft(signal, fs):

    # Trova lo spettro di frequenza del segnale EEG
    fft_result = np.fft.fft(signal)

    # Calcola le frequenze corrispondenti allo spettro di potenza
    n = len(signal)
    frequencies = np.fft.fftfreq(n, 1 / fs)

    # Calcola lo spettro di potenza
    power_spectrum = np.abs(fft_result) ** 2
    return (frequencies, power_spectrum)


def print_AUC(tw_range , AUCtot):
    plt.figure(figsize=(7, 4))
    plt.plot(tw_range, AUCtot, label='AUC')
    plt.title('AUC vs. tw_range')
    plt.xlabel('tw_range')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)
    plt.show()

def print_EER(tw_range, EERtot):
    plt.figure(figsize=(7, 4))
    plt.plot(tw_range, EERtot, label='EER')
    plt.title('EER vs. tw_range')
    plt.xlabel('tw_range')
    plt.ylabel('EER')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_ROC_curve(FAR, FRR, AUC):
    plt.figure(figsize=(5, 5))
    plt.plot(FAR, FRR, label=f'AUC = {AUC:.2f}')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('False Rejection Rate (FRR)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_EER(EER_values):
    plt.plot(range(1, len(EER_values) + 1), EER_values, marker='o')
    plt.xlabel('Numero di Canali')
    plt.ylabel('Valore EER')
    plt.title('EER vs Numero di Canali Selezionati')
    plt.xticks(range(1, len(EER_values) + 1))
    plt.tight_layout()
    plt.show()

def plot_AUC(AUC_values):
    plt.plot(range(1, len(AUC_values) + 1), AUC_values, marker='o')
    plt.xlabel('Numero di Canali')
    plt.ylabel('Valore AUC')
    plt.title('AUC vs Numero di Canali Selezionati')
    plt.xticks(range(1, len(AUC_values) + 1))
    plt.tight_layout()
    plt.show()



def plot_psd_freq_features(frequencies, power_spectrum, indices_of_interest, psd_of_interest):
    plt.semilogy(frequencies, power_spectrum, label='PSD')
    plt.scatter(frequencies[indices_of_interest], psd_of_interest, color='red', label='Frequencies of Interest')
    plt.legend()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.show()



def plot_area_values_psd(features,frequency_intervals):
    plt.bar(range(len(frequency_intervals)), features, tick_label=[str(interval) for interval in frequency_intervals])
    plt.xlabel('Intervallo di Frequenza')
    plt.ylabel('Area sotto il PSD')
    plt.title('Area sotto il PSD per intervallo di frequenza')
    plt.show()

def plot_PSD_area_features(frequencies, power_spectrum,frequency_intervals):
    plt.figure(figsize=(10, 6))
    plt.semilogy(frequencies, power_spectrum, label='Spettro di Potenza', color='blue')

    # Evidenziamo gli intervalli di frequenza di interesse
    for interval in frequency_intervals:
        plt.fill_between(frequencies, 0, power_spectrum,
                         where=((frequencies >= interval[0]) & (frequencies <= interval[1])), alpha=0.3)

    # Aggiungiamo una legenda e le etichette degli assi
    plt.xlabel('Frequenza (Hz)')
    plt.ylabel('Spettro di Potenza')
    plt.title('Spettro di Potenza con Intervalli di Frequenza di Interesse')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_compare(values, vettori_trasposti,titolo,etichette_personalizzate):
    plt.plot(range(1, len(values) + 1), vettori_trasposti, marker='o')
    plt.xlabel('Numero di Canali')
    plt.ylabel('Valori di EER')
    plt.title(titolo)
    plt.xticks(range(1, len(values) + 1))
    plt.legend(etichette_personalizzate)
    plt.show()

