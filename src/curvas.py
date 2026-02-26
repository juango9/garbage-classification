
import matplotlib.pyplot as plt

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