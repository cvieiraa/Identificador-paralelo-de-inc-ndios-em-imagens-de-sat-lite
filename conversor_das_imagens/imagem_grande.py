import numpy as np
from PIL import Image
import tifffile
import math
import os

def gerar_imagem_gigante(caminho_imagem, caminho_saida, target_gb=20):
    target_bytes = target_gb * 1024**3

    # Abre imagem e converte para RGB apenas se necessário
    imagem = Image.open(caminho_imagem)
    print(f"🖼️ Modo original da imagem: {imagem.mode}")
    
    if imagem.mode != "RGB":
        print("⚠️ Convertendo imagem para RGB (modo atual não é RGB)...")
        imagem = imagem.convert("RGB")

    # Converte para numpy array (preserva canais corretamente como RGB)
    tile_array = np.array(imagem, dtype=np.uint8)

    # 🚨 Proteção extra: assegura que canais estão no padrão RGB
    if tile_array.shape[2] == 3:
        r, g, b = tile_array[..., 0], tile_array[..., 1], tile_array[..., 2]
        if np.mean(r) < np.mean(b):
            print("⚠️ Atenção: imagem parece estar em BGR! Corrigindo para RGB...")
            tile_array = tile_array[..., ::-1]  # Inverte os canais (BGR → RGB)

    tile_h, tile_w, channels = tile_array.shape
    tile_bytes = tile_h * tile_w * channels

    print(f"Imagem original: {tile_w}x{tile_h}px, canais: {channels}, ~{tile_bytes / 1024**2:.2f} MB por cópia")

    # Cálculo de quantas réplicas serão necessárias
    tiles_needed = target_bytes / tile_bytes
    n = math.sqrt(tiles_needed)
    n_x = math.ceil(n)
    n_y = math.ceil(tiles_needed / n_x)

    print(f"Objetivo: ~{target_gb} GB → {tiles_needed:.2f} cópias → grade {n_x} x {n_y} = {n_x * n_y} cópias")

    # Replicação da imagem
    linha = np.concatenate([tile_array] * n_x, axis=1)
    final_image = np.concatenate([linha] * n_y, axis=0)

    tamanho_gb = final_image.nbytes / 1024**3
    print(f"Imagem final: {final_image.shape}, tamanho na memória: {tamanho_gb:.2f} GB")

    # ✅ Salva como BigTIFF, preservando os canais RGB
    tifffile.imwrite(caminho_saida, final_image, bigtiff=True)
    print(f"✅ Imagem gigante salva em: {caminho_saida}")
    print(f"📦 Tamanho do arquivo salvo: {os.path.getsize(caminho_saida) / 1024**3:.2f} GB")

if __name__ == "__main__":
    os.makedirs("img_gigantes", exist_ok=True)
    caminho_saida = os.path.join("img_gigantes", "imagem_gigante_20GB.tiff")

    # Altere para o caminho da imagem original com fogo visível
    gerar_imagem_gigante("./img/4.jpg", caminho_saida, target_gb=20)
