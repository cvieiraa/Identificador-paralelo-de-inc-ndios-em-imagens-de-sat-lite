import numpy as np
import tifffile as tiff
import cv2
import time
import os
from tqdm import tqdm

# ==== PARÂMETROS DE DETECÇÃO ====
AREA_MINIMA = 5  # pixels
VERMELHO_BAIXO1 = np.array([0, 120, 120])
VERMELHO_ALTO1 = np.array([10, 255, 255])
VERMELHO_BAIXO2 = np.array([160, 120, 120])
VERMELHO_ALTO2 = np.array([180, 255, 255])

# ==== DETECÇÃO HSV (sem zoom) ====
def detectar_areas_vermelhas_HSV(tile):
    hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)

    # Máscaras para faixas de vermelho
    mask1 = cv2.inRange(hsv, VERMELHO_BAIXO1, VERMELHO_ALTO1)
    mask2 = cv2.inRange(hsv, VERMELHO_BAIXO2, VERMELHO_ALTO2)
    mascara = cv2.bitwise_or(mask1, mask2)

    # Morfologia
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centros = []
    for contorno in contornos:
        if cv2.contourArea(contorno) >= AREA_MINIMA:
            M = cv2.moments(contorno)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centros.append((cx, cy))
    return centros

# ==== PROCESSAMENTO SEQUENCIAL ====
def process_image_sequencial(image_path, output_path="saida_hsv_sequencial.jpg"):
    print(f"🖼️ Lendo imagem: {image_path}")
    ext = os.path.splitext(image_path)[1].lower()

    if ext in [".tif", ".tiff"]:
        with tiff.TiffFile(image_path) as tif:
            img = tif.asarray()
    else:
        img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Imagem inválida: {image_path}")

    # Garantir 3 canais
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    print("🔥 Detectando focos...")
    start = time.time()

    centros = detectar_areas_vermelhas_HSV(img)

    for cx, cy in centros:
        cv2.circle(img, (cx, cy), 6, (0, 0, 255), 2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)

    elapsed = time.time() - start
    print(f"✅ {len(centros)} focos detectados em {elapsed:.2f} segundos.")
    print(f"💾 Imagem salva em: {output_path}")

    return len(centros), elapsed

# ==== EXECUÇÃO PRINCIPAL ====
if __name__ == "__main__":
    image_to_process = "D:/fire_detector/fire_detector/img/4.jpg" # Altere aqui para o caminho da imagem
    output_image_name = "resultados_hsv/saida_hsv_sequencial.jpg"

    try:
        focos, tempo = process_image_sequencial(image_to_process, output_image_name)
        print(f"📂 Imagem de saída salva em: {output_image_name}")
    except FileNotFoundError as e:
        print(f"❌ Erro: {e}")
    except Exception as e:
        print(f"❌ Erro durante o processamento: {e}")
