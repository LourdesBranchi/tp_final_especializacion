import numpy as np
from PIL import Image


def resize_con_padding(arr, size=256, es_mascara=False):
    """
    Redimensiona a un cuadrado de `size`x`size` SIN distorsionar la relación
    de aspecto: escala manteniendo proporciones y agrega padding (relleno)
    en vez de estirar la imagen.
    """
    alto, ancho = arr.shape
    escala = size / max(alto, ancho)
    nuevo_alto = int(round(alto * escala))
    nuevo_ancho = int(round(ancho * escala))

    if es_mascara:
        # PIL no soporta arrays int64 (el error que viste: "Cannot handle
        # this data type: (1, 1), <i8"). Las máscaras solo tienen valores
        # 0-3, así que uint8 alcanza de sobra para el resize -- el cast a
        # int64 final para PyTorch se hace después, en CamusDataset.
        arr_para_resize = arr.astype(np.uint8)
        interpolacion = Image.NEAREST
    else:
        arr_para_resize = arr
        interpolacion = Image.BILINEAR

    img_resized = np.array(
        Image.fromarray(arr_para_resize).resize((nuevo_ancho, nuevo_alto), interpolacion)
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