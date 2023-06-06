import matplotlib.pyplot as plt
import pyautogui


# Definisci il numero di righe e colonne
num_rows = 6
num_cols = 8


# Crea la griglia di subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 10), sharey=True)

# # Opzionalmente, nascondi gli assi per una visualizzazione più pulita
# for ax in axes.flatten():
#     ax.axis('off')
fig.subplots_adjust(top=0.97, bottom=0.055, left=0.045, right=0.97, hspace=0.41, wspace=0.21)
# Mostra il grafico
fig.savefig('nome_immagine.png')
plt.show()
# Chiudi la figura
plt.close(fig)
# import matplotlib.pyplot as plt

# # Definisci il numero di righe e colonne
# num_rows = 6
# num_cols = 8

# # Crea la griglia di subplots
# fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10), sharey=True)

# # Imposta i margini tra i subplots
# fig.subplots_adjust(left=0.05, right=0.97, top=0.97, bottom=0.05, wspace=0.2, hspace=0.2)

# # Ottieni l'oggetto FigureManager
# manager = plt.get_current_fig_manager()

# # Passa alla modalità schermo intero
# manager.full_screen_toggle()

# # Mostra il grafico
# plt.show()
