def aplicar_semilla(seed):
    import random
    import numpy as np
    import tensorflow as tf
    import os
    
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)