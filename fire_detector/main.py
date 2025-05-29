import cv2
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor

PASTA_IMAGENS = "imagens"
PASTA_RESULTADOS = "resultados"
os.makedirs(PASTA_RESULTADOS, exist_ok=True)

# Faixas HSV para tons de vermelho puro (ignora laranja e amarelo)
VERMELHO_BAIXO1 = np.array([0, 100, 100])
VERMELHO_ALTO1 = np.array([10, 255, 255])
VERMELHO_BAIXO2 = np.array([170, 100, 100])
VERMELHO_ALTO2 = np.array([179, 255, 255])

def detectar_fogo_vermelho(imagem_path):
    imagem_bgr = cv2.imread(imagem_path)
    if imagem_bgr is None:
        print(f"[ERRO] Imagem inválida: {imagem_path}")
        return 0

    imagem_hsv = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV)

    # Máscaras para tons de vermelho
    mascara1 = cv2.inRange(imagem_hsv, VERMELHO_BAIXO1, VERMELHO_ALTO1)
    mascara2 = cv2.inRange(imagem_hsv, VERMELHO_BAIXO2, VERMELHO_ALTO2)
    mascara_fogo = cv2.bitwise_or(mascara1, mascara2)

    total_fogo = cv2.countNonZero(mascara_fogo)

    # Destacar os focos
    resultado = cv2.bitwise_and(imagem_bgr, imagem_bgr, mask=mascara_fogo)
    nome_arquivo = os.path.basename(imagem_path)
    caminho_saida = os.path.join(PASTA_RESULTADOS, f"vermelho_{nome_arquivo}")
    cv2.imwrite(caminho_saida, resultado)

    print(f"[INFO] {nome_arquivo}: {total_fogo} focos de incêndio detectados.")
    return total_fogo

def main():
    imagens = [
        os.path.join(PASTA_IMAGENS, f)
        for f in os.listdir(PASTA_IMAGENS)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    with ProcessPoolExecutor() as executor:
        resultados = list(executor.map(detectar_fogo_vermelho, imagens))

    total_geral = sum(resultados)
    print(f"\n🔥 Total geral de focos de incêndio detectados aproximadamente: {total_geral}")

if __name__ == "__main__":
    main()
