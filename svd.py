import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from PIL import Image
import math 

# Parámetros
MAX_RANK = 100
# Imagen 
FNAME = 'Gauss_1828.jpg'

image = Image.open(FNAME).convert("L")
img_mat = np.asarray(image)

print(img_mat.shape)

# Descomposición en valores singulares 
U, s, V = np.linalg.svd(img_mat, full_matrices=True)

s = np.diag(s)

# Vamos obteniendo los valores singulares 

def k_grados_aproximacion(k, show = True): 
    global U, s, V 
    approx = U[:, :k] @ s[0:k, :k] @ V[:k, :]

    E = img_mat - approx
    err = linalg.norm(E) / linalg.norm(img_mat)

    #print(approx.shape)
    if(show): 
        print(f"Error para la aproximación con {k} grados es de: {err}")
        img = plt.imshow(approx, cmap='gray')
        plt.title(f'SVD con grados de aproximación:  {k}')
        plt.plot()
        plt.show()
    return err 

k_grados_aproximacion(10)
k_grados_aproximacion(20)
k_grados_aproximacion(40)
k_grados_aproximacion(60)

errores = []
for i in range(MAX_RANK): 
    errores.append(k_grados_aproximacion(i, False))

plt.plot(errores)
plt.title("Errores en la aproximación ")
plt.show()

vsing = []
i = 0 
for val in s: 
    v = math.log(val[i])
    vsing.append(v)
    i+= 1

plt.plot(vsing,[i for i in range(i)])
plt.title("Escala de valores singulares (log)")
plt.show()