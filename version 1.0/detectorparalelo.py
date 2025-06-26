import numpy as np
import tifffile as tiff
import cv2
import time
import os
from skimage.measure import label, regionprops
from tqdm import tqdm
import multiprocessing

def preprocess_tile_for_fire(tile):
    if tile.shape[2] == 4:
        tile = cv2.cvtColor(tile, cv2.COLOR_BGRA2BGR)

    r = tile[:, :, 2].astype(np.int16)
    g = tile[:, :, 1].astype(np.int16)
    b = tile[:, :, 0].astype(np.int16)

    red_mask = (r > 180) & ((r - g) > 80) & ((r - b) > 80)

    binary = np.zeros(tile.shape[:2], dtype=np.uint8)
    binary[red_mask] = 255

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.dilate(binary, kernel, iterations=1)

    return binary

def get_fire_centroids(tile, area_minima=1):
    binary = preprocess_tile_for_fire(tile)
    labeled = label(binary)
    props = regionprops(labeled)
    centroids = [region.centroid for region in props if region.area >= area_minima]
    return centroids

def process_tile_wrapper(args):
    tile, x, y = args
    centroids = get_fire_centroids(tile, area_minima=10)
    return x, y, centroids

def process_large_image_parallel(image_path, tile_size=1024, output_path="resultados_paralelo/resultado_incendios_paralelo.tiff", num_threads=4, chunk_size=10):
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
    print(f"⚙️ Usando {num_threads} threads para processamento paralelo.")

    tiles_data = []
    ys = list(range(0, height, tile_size))
    xs = list(range(0, width, tile_size))
    for y in ys:
        for x in xs:
            tile = img[y:min(y+tile_size, height), x:min(x+tile_size, width)]
            tiles_data.append((tile, x, y))

    output = np.zeros((height, width, 3), dtype=np.uint8)
    total_fires_detected = 0

    print("▶️ Processando tiles (paralelo)...")

    with multiprocessing.Pool(processes=num_threads) as pool:
        results = pool.imap_unordered(process_tile_wrapper, tiles_data, chunksize=chunk_size)

        for x_coord, y_coord, centroids in tqdm(results, total=len(tiles_data), desc="Tiles Processados", unit="tile"):
            total_fires_detected += len(centroids)
            for cy, cx in centroids:
                cv2.circle(output, (int(cx)+x_coord, int(cy)+y_coord), 6, (0, 0, 255), 2)

    # Garante que a pasta de saída exista
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Salva imagem final
    cv2.imwrite(output_path, output)
    elapsed = time.time() - start
    print(f"✅ Processamento Paralelo Concluído: {total_fires_detected} focos detectados em {elapsed:.2f} segundos.")
    return elapsed, total_fires_detected

if __name__ == "__main__":
    image_to_process = "D:/fire_detector/fire_detector/img/4.jpg"  # <-- Caminho da imagem
    output_image_name_parallel = "resultados_paralelo/resultado_incendios_paralelo.tiff"
    tile_dimension = 1024
    num_parallel_threads = 12  # Ajuste para 2, 4, 8, etc.

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