import torch
import torch.nn as nn
import torch.nn.functional as F


class ComboLoss(nn.Module):
    """
    Combo Loss (Taghanaki et al., 2019) para segmentación multi-clase.

    Combina dos términos:
    - Dice: mide solapamiento espacial global. Ataca el "input imbalance"
      (el fondo domina en cantidad de píxeles frente a LV/MYO/LA).
    - Cross-Entropy modificada, ponderada por beta: penaliza distinto los
      falsos negativos que los falsos positivos. Ataca el "output imbalance".

    alpha: peso entre CE y Dice (0.5 = el valor que los autores recomiendan
           como mejor promedio general).
    beta: controla el trade-off FP/FN dentro de la CE modificada.
          beta alto -> penaliza más perder estructura (falsos negativos).
          Los autores probaron esto específicamente en un dataset de
          ULTRASOUND y encontraron beta=0.7 como el mejor valor ahí,
          por eso es el default acá en vez de un 0.5 neutro.
    class_weights: pesos opcionales por clase (tensor de tamaño num_clases)
          para atacar además el desbalance estructural fondo/LV/MYO/LA.
          Si es None, todas las clases pesan igual.
    """

    def __init__(self, alpha=0.5, beta=0.7, class_weights=None, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.class_weights = class_weights
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: [B, C, H, W] (salida cruda del modelo, sin softmax)
        # targets: [B, H, W] (máscara con clase entera 0..C-1 por píxel)
        num_clases = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_clases).permute(0, 3, 1, 2).float()

        # --- Término Dice (promedio entre las 4 clases) ---
        dims = (0, 2, 3)
        interseccion = (probs * targets_onehot).sum(dims)
        union = probs.sum(dims) + targets_onehot.sum(dims)
        dice_por_clase = (2 * interseccion + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_por_clase.mean()

        # --- Término Cross-Entropy modificada, ponderada por beta ---
        probs_clamp = torch.clamp(probs, self.smooth, 1 - self.smooth)
        castigo_falsos_negativos = targets_onehot * torch.log(probs_clamp)
        castigo_falsos_positivos = (1 - targets_onehot) * torch.log(1 - probs_clamp)
        ce_termino = -(self.beta * castigo_falsos_negativos
                        + (1 - self.beta) * castigo_falsos_positivos)

        if self.class_weights is not None:
            pesos = self.class_weights.view(1, num_clases, 1, 1).to(logits.device)
            ce_termino = ce_termino * pesos

        ce_loss = ce_termino.mean()

        return self.alpha * ce_loss + (1 - self.alpha) * dice_loss