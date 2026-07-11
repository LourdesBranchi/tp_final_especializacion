import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def crear_unet_resnet34(num_clases=4):
    return smp.Unet(encoder_name='resnet34', encoder_weights='imagenet',
                     in_channels=1, classes=num_clases)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (Hu et al., 2018): recalibra la importancia
    de cada canal de features. Este es un componente que ResDUnet le agrega al
    U-Net clásico."""
    def __init__(self, canales, reduccion=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(canales, canales // reduccion), nn.ReLU(inplace=True),
            nn.Linear(canales // reduccion, canales), nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        peso = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * peso


class BloqueResidualSE(nn.Module):
    """Bloque de ResDUnet: dos convoluciones 3x3 con conexión residual
    (facilita el flujo de gradiente) + una unidad SE al final."""
    def __init__(self, canales_in, canales_out):
        super().__init__()
        self.conv1 = nn.Conv2d(canales_in, canales_out, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(canales_out)
        self.conv2 = nn.Conv2d(canales_out, canales_out, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(canales_out)
        self.se = SEBlock(canales_out)
        self.relu = nn.ReLU(inplace=True)
        self.proyeccion = (nn.Conv2d(canales_in, canales_out, 1)
                            if canales_in != canales_out else nn.Identity())

    def forward(self, x):
        residual = self.proyeccion(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.relu(out + residual)


class ResDUnet(nn.Module):
    """
    Versión simplificada de ResDUnet (Amer et al., 2021): U-Net con
    bloques residuales + SE en encoder y decoder.

    Ojo: no es una réplica exacta del paper, sino que esto
    respeta la idea arquitectónica central (residual + SE) pero con
    canales "estándar" de U-Net. La idea no es replicar la arquitectura exacta, 
    sino "inspirarnos" en ella
    """
    def __init__(self, num_clases=4, canales_base=32):
        super().__init__()
        c = canales_base
        self.enc1 = BloqueResidualSE(1, c)
        self.enc2 = BloqueResidualSE(c, c * 2)
        self.enc3 = BloqueResidualSE(c * 2, c * 4)
        self.enc4 = BloqueResidualSE(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = BloqueResidualSE(c * 8, c * 16)

        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = BloqueResidualSE(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = BloqueResidualSE(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = BloqueResidualSE(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = BloqueResidualSE(c * 2, c)

        self.salida = nn.Conv2d(c, num_clases, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.salida(d1)


def crear_modelo(nombre_arquitectura, num_clases=4):
    """Factory simple para instanciar por nombre desde el notebook."""
    if nombre_arquitectura == 'unet_resnet34':
        return crear_unet_resnet34(num_clases)
    elif nombre_arquitectura == 'resdunet':
        return ResDUnet(num_clases=num_clases)
    raise ValueError(f"Arquitectura desconocida: {nombre_arquitectura}")