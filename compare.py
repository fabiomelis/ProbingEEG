import selection_alg
import core
import numpy as np
import matplotlib.pyplot as plt
import utils
import os

def compare_all_clips(matrix_4D, tw, fs):

    n_sbjs, n_clips, n_channels, n_samples = matrix_4D.shape

    vettori = []
    stringhe_di_stampa = []

    for clip in range(n_clips):
        reduced_data = matrix_4D[:, clip, :, :]

        print('Clip analizzata: ', clip + 1)

        values, channel_list = selection_alg.forward_selection_eer(n_channels, reduced_data, tw, fs, 'welch')
        #values, channel_list = selection_alg.forward_selection_auc(n_channels, reduced_data, tw, fs, 'off')
        #values, channel_list = selection_alg.backward_selection_eer(n_channels, reduced_data, tw, fs, 'off')
        #values, channel_list = selection_alg.backward_selection_auc(n_channels, reduced_data, tw, fs, 'off')


        vettori.append(values)
        stringhe_di_stampa.append(f'Clip {clip + 1}: {channel_list}')

    # Trasponi la lista di vettori per ottenere una lista di colonne invece di righe
    vettori_trasposti = np.array(vettori).T

    print('\n'.join(stringhe_di_stampa))

    # Crea il grafico con tutti i vettori sovrapposti
    plt.plot(range(1, len(values) + 1), vettori_trasposti, marker='o')
    plt.xlabel('Numero di Canali')
    plt.ylabel('Valori di EER')
    plt.title('Valori di EER sovrapposti per ogni Clip')
    plt.xticks(range(1, len(values) + 1))
    plt.show()




def compare_forward_with_list(matrix_4D, tw, fs):

    n_sbjs, n_clips, n_channels, n_samples = matrix_4D.shape

    vettori = []
    stringhe_di_stampa = []
    output_folder = r'C:\Users\fabio\Desktop\ING INFORMATICA\TESI\img compare'

    for clip in range(n_clips):
        reduced_data = matrix_4D[:, clip, :, :]

        print('Clip analizzata: ', clip + 1)

        values, channel_list = selection_alg.forward_selection_eer(n_channels, reduced_data, tw, fs, 'off')

        vettori.append(values)
        stringhe_di_stampa.append(f'Clip {clip + 1}: {channel_list}')

        list_channel = [6, 1, 2, 5, 14, 8, 13, 3, 9, 11, 10, 7, 12, 4]
        values, channel_list = core.compute_EER_fixed_list(list_channel, reduced_data, tw, fs, 'off')

        vettori.append(values)
        stringhe_di_stampa.append(f'Clip {clip + 1}: {channel_list}')

        # Trasponi la lista di vettori per ottenere una lista di colonne invece di righe
        vettori_trasposti = np.array(vettori).T

        print('\n'.join(stringhe_di_stampa))

        # Crea il grafico con tutti i vettori sovrapposti
        plt.plot(range(1, len(values) + 1), vettori_trasposti, marker='o')
        plt.xlabel('Numero di Canali')
        plt.ylabel('Valori di EER')
        plt.title('Valori di EER sovrapposti')
        plt.xticks(range(1, len(values) + 1))
        plt.legend(['Forward Selection', 'Lista Canali Fissati'])
        plt.savefig(os.path.join(output_folder, f'grafico_clip_{clip + 1}.png'))
        plt.close()



def compare_all_FOOOF_features(dataset, tw, fs):

    n_sbjs, n_channels, n_samples = dataset.shape

    vettori = []
    stringhe_di_stampa = []

    string = 'exp'
    EER_values , channel_list = selection_alg.forward_selection_eer(n_channels, dataset, tw, fs, string)

    vettori.append(EER_values)
    stringhe_di_stampa.append(f'FOOOF features = EXP: {channel_list}')

    string = 'off'
    EER_values, channel_list = selection_alg.forward_selection_eer(n_channels, dataset, tw, fs, string)

    vettori.append(EER_values)
    stringhe_di_stampa.append(f'FOOOF features = EXP, OFF: {channel_list}')

    string = 'freq'
    EER_values, channel_list = selection_alg.forward_selection_eer(n_channels, dataset, tw, fs, string)

    vettori.append(EER_values)
    stringhe_di_stampa.append(f'FOOOF features = EXP, OFF, FREQ: {channel_list}')

    # Trasponi la lista di vettori per ottenere una lista di colonne invece di righe
    vettori_trasposti = np.array(vettori).T

    print('\n'.join(stringhe_di_stampa))

    # Crea il grafico con tutti i vettori sovrapposti
    plt.plot(range(1, len(EER_values) + 1), vettori_trasposti, marker='o')

    # Aggiungi etichette e titolo al grafico
    plt.xlabel('Numero di Canali')
    plt.ylabel('Valori di EER')
    plt.title('Valori di EER sovrapposti per Features estratte')
    plt.xticks(range(1, len(EER_values) + 1))
    # Mostra il grafico
    plt.show()