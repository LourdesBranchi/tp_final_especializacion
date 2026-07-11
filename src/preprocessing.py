import numpy as np
from PIL import Image


def resize_con_padding(arr, size=256, es_mascara=False):
    """
    Redimensiona a un cuadrado de `size`x`size` SIN distorsionar la relación
    de aspecto: escala manteniendo proporciones y agrega padding (relleno)
    en vez de estirar la imagen.

    Por qué: el EDA de dimensiones mostró que la relación de aspecto real de
    CAMUS es mayoritariamente ~1.2, no 1:1. Un resize directo a 256x256
    (lo que hacía VPC II) distorsiona la geometría de la mayoría de las
    imágenes y máscaras.

    es_mascara: True para máscaras de segmentación (usa interpolación
                'nearest' para no promediar valores de clase, y rellena
                con 0 = clase fondo). False para imágenes (interpolación
                bilineal, rellena con negro).
    """
    alto, ancho = arr.shape
    escala = size / max(alto, ancho)
    nuevo_alto = int(round(alto * escala))
    nuevo_ancho = int(round(ancho * escala))

    interpolacion = Image.NEAREST if es_mascara else Image.BILINEAR
    img_resized = np.array(
        Image.fromarray(arr).resize((nuevo_ancho, nuevo_alto), interpolacion)
    )

    pad_alto = size - nuevo_alto
    pad_ancho = size - nuevo_ancho
    top, bottom = pad_alto // 2, pad_alto - pad_alto // 2
    left, right = pad_ancho // 2, pad_ancho - pad_ancho // 2

    return np.pad(img_resized, ((top, bottom), (left, right)),
                   mode='constant', constant_values=0)


def normalizar_minmax(arr):
    """Normalización min-max por imagen a [0,1] (heredado de VPC II — tu
    EDA de intensidades confirmó que esta elección, por imagen y no global,
    sigue siendo razonable)."""
    minimo, maximo = arr.min(), arr.max()
    if maximo - minimo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - minimo) / (maximo - minimo)