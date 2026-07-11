# src/xai_utils.py (a completar cuando haya modelo entrenado)

def seg_grad_cam(modelo, imagen, clase_objetivo, capa_conv):
    """
    imagen: tensor [1, 1, H, W]
    clase_objetivo: índice de clase (1=LV, 2=MYO, 3=LA)
    capa_conv: la capa convolucional del modelo sobre la que se calcula el heatmap
               (normalmente cerca del bottleneck o antes de la capa final)
    """
    activaciones = {}
    gradientes = {}

    def guardar_activacion(module, entrada, salida):
        activaciones['valor'] = salida
    def guardar_gradiente(module, grad_entrada, grad_salida):
        gradientes['valor'] = grad_salida[0]

    capa_conv.register_forward_hook(guardar_activacion)
    capa_conv.register_full_backward_hook(guardar_gradiente)

    salida = modelo(imagen)
    score = salida[:, clase_objetivo, :, :].sum()  # suma sobre los píxeles de la clase de interés
    modelo.zero_grad()
    score.backward()

    pesos = gradientes['valor'].mean(dim=(2, 3), keepdim=True)  # promedio global del gradiente
    heatmap = (pesos * activaciones['valor']).sum(dim=1).relu()
    return heatmap.squeeze().detach().cpu().numpy()