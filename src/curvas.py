
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def graf_pedida(hist):
   plt.plot(hist.history['loss'],label="loss")
   plt.plot(hist.history['val_loss'],label="val_loss")
   plt.legend()
   grafico =plt.show()
   return grafico

def graf_acc(hist,):
    plt.plot(hist.history['sparse_categorical_accuracy'],label="acc")
    plt.plot(hist.history['val_sparse_categorical_accuracy'],label="val_acc")
    plt.legend()
    grafico=plt.show()
    return grafico

def plot_confusion(y_true, y_pred, class_names, title='Matriz de confusion'):
    # Redondear predicciones a etiquetas enteras
    y_pred_labels = np.rint(y_pred).astype(int)
    y_pred_labels = np.clip(y_pred_labels, 0, len(class_names) - 1)

    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred_labels, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues')
    plt.title(title)
    plt.tight_layout()
    plt.show()