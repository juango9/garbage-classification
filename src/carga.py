import pandas as pd
import os
def cargar_datos(ruta_archivo):
    
    data={}
    for i in os.listdir(ruta_archivo):
        for j in os.walk(os.path.join(ruta_archivo, i)):
            for k in j[2]:
                data[os.path.join(ruta_archivo, i, k)]=i

    df=pd.DataFrame(data.items(),columns=['x','y'])
    df=df.sample(frac=1) 
    return df