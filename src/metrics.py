import numpy as np


def dice_por_clase(pred, target, num_clases=4, smooth=1e-6):
    """pred, target: tensores [H, W] con clase entera (no probabilidades).
    Devuelve array de Dice por clase: [fondo, LV, MYO, LA]."""
    dices = []
    for c in range(num_clases):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        interseccion = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dices.append(((2 * interseccion + smooth) / (union + smooth)).item())
    return np.array(dices)


def iou_por_clase(pred, target, num_clases=4, smooth=1e-6):
    ious = []
    for c in range(num_clases):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        interseccion = (pred_c * target_c).sum()
        union = ((pred_c + target_c) > 0).float().sum()
        ious.append(((interseccion + smooth) / (union + smooth)).item())
    return np.array(ious)