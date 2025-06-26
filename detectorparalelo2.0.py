import numpy as np
import tifffile as tiff
import cv2
import time
import os
from tqdm import tqdm
import multiprocessing

# ==== CONFIGURAÇÕES HSV E DETECÇÃO ====
AREA_MINIMA = 1  # pixels

def detectar_areas_vermelhas_tile_RGB(tile, zoom=2):
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

# ==== VARIÁVEL GLOBAL DO MEMMAP ====
global_img = None

def init_worker(img_path):
    global global_img
    global_img = tiff.memmap(img_path)  # cada processo carrega sua própria referência memmap

# ==== PROCESSA UM TILE DADO SEU X, Y ====
def process_tile_wrapper(args):
    x, y, tile_size, height, width = args
    tile = global_img[y:min(y+tile_size, height), x:min(x+tile_size, width)]

    # Garantir 3 canais
    if tile.ndim == 2:
        tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
    elif tile.shape[2] == 4:
        tile = cv2.cvtColor(tile, cv2.COLOR_BGRA2BGR)

    centros = detectar_areas_vermelhas_tile_RGB(tile)
    return x, y, centros

# ==== PROCESSAMENTO PARALELO ====
def process_large_image_parallel(image_path, tile_size=1024, output_path="resultados_paralelo/resultado_incendios_paralelo.tiff", num_threads=4, chunk_size=10):
    print(f"🔄 Carregando imagem (modo leitura por tile): {image_path}")
    start = time.time()

    with tiff.TiffFile(image_path) as tif:
        page = tif.pages[0]
        height, width = page.shape[:2]

    print(f"📐 Dimensões da imagem: {width}x{height}")
    print(f"⚙️ Usando {num_threads} threads para processamento paralelo.")

    # Lista apenas de posições
    tiles_data = [(x, y, tile_size, height, width)
                  for y in range(0, height, tile_size)
                  for x in range(0, width, tile_size)]

    output = np.zeros((height, width, 3), dtype=np.uint8)
    total_detectados = 0

    print("▶️ Processando tiles em paralelo...")
    with multiprocessing.Pool(processes=num_threads, initializer=init_worker, initargs=(image_path,)) as pool:
        results = pool.imap_unordered(process_tile_wrapper, tiles_data, chunksize=chunk_size)

        for x_coord, y_coord, centros in tqdm(results, total=len(tiles_data), desc="Tiles Processados", unit="tile"):
            total_detectados += len(centros)
            for cx, cy in centros:
                cv2.circle(output, (cx + x_coord, cy + y_coord), 6, (0, 0, 255), 2)

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_path, output)

    elapsed = time.time() - start
    print(f"✅ Processamento Paralelo Concluído: {total_detectados} focos detectados em {elapsed:.2f} segundos.")
    return elapsed, total_detectados

# ==== EXECUÇÃO ====
if __name__ == "__main__":
    image_to_process = "D:/fire_detector/fire_detector/img/4.jpg"
    output_image_name_parallel = "resultados_paralelo/resultado_incendios_paralelo.tiff"
    tile_dimension = 1024
    num_parallel_threads = 6  # Ajuste conforme sua máquina

    try:
        elapsed, fires = process_large_image_parallel(
            image_to_process,
            tile_size=tile_dimension,
            output_path=output_image_name_parallel,
            num_threads=num_parallel_threads,
            chunk_size=10
        )
        print(f"📂 Imagem salva em: {output_image_name_parallel}")
    except FileNotFoundError as e:
        print(f"❌ Erro: {e}. Verifique o caminho da imagem.")
    except Exception as e:
        print(f"❌ Erro durante o processamento: {e}")
