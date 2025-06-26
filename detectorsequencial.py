import numpy as np
import tifffile as tiff
import cv2
import time
import os
from tqdm import tqdm

AREA_MINIMA = 1  # igual ao paralelo
ZOOM = 2          # também igual

def detectar_areas_vermelhas_tile_RGB(tile, zoom=ZOOM):
    imagem_ampliada = cv2.resize(tile, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    R, G, B = cv2.split(imagem_ampliada)
    mascara = (R > 150) & (G < 100) & (B < 100)
    mascara = mascara.astype(np.uint8) * 255
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centros = []
    for contorno in contornos:
        if cv2.contourArea(contorno) >= AREA_MINIMA:
            M = cv2.moments(contorno)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]) // zoom
                cy = int(M["m01"] / M["m00"]) // zoom
                centros.append((cx, cy))
    return centros

def process_large_image_sequential(image_path, tile_size=1024, output_path="resultados_sequencial/resultado_incendios_sequencial.tiff", delay_per_tile_seconds=0):
    print(f"🔄 Carregando imagem: {image_path}")
    start = time.time()

    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".tif", ".tiff"]:
        with tiff.TiffFile(image_path) as tif:
            page = tif.pages[0]
            height, width = page.shape[:2]
            img = tif.asarray(out='memmap')
    else:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        height, width = img.shape[:2]

    print(f"📐 Dimensões da imagem: {width}x{height}")

    tiles = []
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile = img[y:min(y+tile_size, height), x:min(x+tile_size, width)]
            tiles.append((tile, x, y))

    output = np.zeros((height, width, 3), dtype=np.uint8)
    total_detectados = 0

    print("▶️ Processando tiles sequencialmente...")
    for tile, x, y in tqdm(tiles, desc="Tiles", unit="tile"):
        centros = detectar_areas_vermelhas_tile_RGB(tile)
        total_detectados += len(centros)
        for cx, cy in centros:
            cv2.circle(output, (cx + x, cy + y), 6, (0, 0, 255), 2)
        time.sleep(delay_per_tile_seconds)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, output)

    elapsed = time.time() - start
    print(f"✅ Concluído: {total_detectados} focos detectados em {elapsed:.2f} segundos.")
    return elapsed, total_detectados

# Execução
if __name__ == "__main__":
    image_to_process = "D:/fire_detector/fire_detector/img/4.jpg"
    output_image_name = "resultados_sequencial/resultado_incendios_sequencial.tiff"
    tile_dimension = 1024
    delay = 0

    try:
        process_large_image_sequential(image_to_process, tile_size=tile_dimension, output_path=output_image_name, delay_per_tile_seconds=delay)
        print(f"📂 Imagem salva em: {output_image_name}")
    except Exception as e:
        print(f"❌ Erro: {e}")
