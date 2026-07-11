import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk

from src.preprocessing import resize_con_padding, normalizar_minmax


def parsear_cfg(path):
    """Lee un .cfg de CAMUS y devuelve sus campos como diccionario."""
    datos = {}
    with open(path, 'r') as f:
        for linea in f:
            linea = linea.strip()
            if ':' in linea:
                clave, valor = linea.split(':', 1)
                datos[clave.strip()] = valor.strip()
    return datos


def cargar_metadata(carpeta_base='datos_corazon/database_nifti'):
    """Arma el DataFrame con patient_id, sex, age, EF, image_quality
    (mismo criterio que usamos en el EDA)."""
    pacientes = sorted([p for p in os.listdir(carpeta_base) if p.startswith('patient')])
    registros = []
    for pid in pacientes:
        cfg_path = os.path.join(carpeta_base, pid, 'Info_2CH.cfg')
        if not os.path.exists(cfg_path):
            continue
        datos = parsear_cfg(cfg_path)
        registros.append({
            'patient_id': pid,
            'sex': datos.get('Sex'),
            'age': int(datos.get('Age')),
            'EF': int(datos.get('EF')),
            'image_quality': datos.get('ImageQuality'),
        })
    return pd.DataFrame(registros)


def particionar_pacientes(df_metadata, n_val_por_sexo=20, n_test_por_sexo=20, seed=42):
    """
    Partición a nivel de PACIENTE (nunca de imagen, para evitar data leakage).

    Por qué no es un simple 70/15/15 proporcional:
    F es el recurso escaso (170 en total). Si val/test se armaran como
    fracciones proporcionales del dataset completo (ej. 15% cada uno),
    se consumiría una fracción desproporcionada de las pocas mujeres
    disponibles, dejando muy pocas para armar después un conjunto de
    entrenamiento balanceado.

    Por eso: primero se reservan val y test con un número FIJO e IGUAL de
    pacientes por sexo -- así ambos quedan balanceados sin vaciar la
    reserva de mujeres. Lo que sobra ("pool de entrenamiento") se usa para:
        - train_imbalanced: el pool completo, tal cual (composición natural)
        - train_balanced: mismo pool, undersampleando M para igualar a F
    """
    rng_seed = seed
    val_ids, test_ids = [], []

    for sexo in ['F', 'M']:
        ids_sexo = df_metadata[df_metadata.sex == sexo]['patient_id'].tolist()
        ids_sexo = shuffle(ids_sexo, random_state=rng_seed)
        val_ids += ids_sexo[:n_val_por_sexo]
        test_ids += ids_sexo[n_val_por_sexo:n_val_por_sexo + n_test_por_sexo]

    pool = df_metadata[~df_metadata.patient_id.isin(val_ids + test_ids)]

    train_imbalanced_ids = pool['patient_id'].tolist()

    pool_f = pool[pool.sex == 'F']['patient_id'].tolist()
    pool_m = pool[pool.sex == 'M']['patient_id'].tolist()
    n_balanceado = min(len(pool_f), len(pool_m))
    pool_m_balanceado = shuffle(pool_m, random_state=rng_seed)[:n_balanceado]
    train_balanced_ids = pool_f + pool_m_balanceado

    return {
        'val': val_ids,
        'test': test_ids,
        'train_imbalanced': train_imbalanced_ids,
        'train_balanced': train_balanced_ids,
    }


def resumen_particiones(particiones, df_metadata):
    """Imprime la composición de sexo de cada partición -- para verificar
    que el diseño se cumplió como se esperaba antes de entrenar nada."""
    for nombre, ids in particiones.items():
        sub = df_metadata[df_metadata.patient_id.isin(ids)]
        conteo = sub['sex'].value_counts().to_dict()
        print(f"{nombre}: {len(ids)} pacientes -> {conteo}")


class CamusDataset(Dataset):
    """
    Dataset de PyTorch para CAMUS. Cada paciente aporta hasta 4 imágenes
    (2CH/4CH x ED/ES), tratadas como ejemplos independientes -- mismo
    enfoque que Leclerc et al. (2019) y que VPC II: un solo modelo
    entrenado sobre todas las vistas/fases combinadas.
    """
    VISTAS = ['2CH', '4CH']
    FASES = ['ED', 'ES']

    def __init__(self, patient_ids, carpeta_base='datos_corazon/database_nifti',
                 size=256, augmentar=False):
        self.carpeta_base = carpeta_base
        self.size = size
        self.augmentar = augmentar
        self.muestras = self._listar_muestras(patient_ids)

    def _listar_muestras(self, patient_ids):
        muestras = []
        for pid in patient_ids:
            for vista in self.VISTAS:
                for fase in self.FASES:
                    img_path = os.path.join(self.carpeta_base, pid, f'{pid}_{vista}_{fase}.nii.gz')
                    mask_path = os.path.join(self.carpeta_base, pid, f'{pid}_{vista}_{fase}_gt.nii.gz')
                    if os.path.exists(img_path) and os.path.exists(mask_path):
                        muestras.append((img_path, mask_path))
        return muestras

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, idx):
        img_path, mask_path = self.muestras[idx]

        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype(np.float32)
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).astype(np.int64)

        img = resize_con_padding(img, size=self.size, es_mascara=False)
        mask = resize_con_padding(mask, size=self.size, es_mascara=True)
        img = normalizar_minmax(img)

        # TODO: augmentation conservador (rotación leve, zoom, brillo/contraste,
        # ruido gaussiano, SIN flips) -- lo sumamos cuando lleguemos a esa etapa

        img_tensor = torch.from_numpy(img).unsqueeze(0).float()   # [1, H, W]
        mask_tensor = torch.from_numpy(mask).long()                # [H, W]
        return img_tensor, mask_tensor