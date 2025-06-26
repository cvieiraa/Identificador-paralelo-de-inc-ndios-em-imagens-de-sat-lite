from mpi4py import MPI
import os
import rasterio
from rasterio.windows import Window
import numpy as np
import cv2
import time
import warnings

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


# CONTADOR DE TEMPO (INÍCIO)

inicio = time.time()


# CONFIGURAÇÕES

PASTA_IMAGENS = "incendios"
PASTA_RESULTADOS = "resultados"
os.makedirs(PASTA_RESULTADOS, exist_ok=True)

TILE_SIZE = 5000  # Tamanho dos blocos (tiles)

VERMELHO_BAIXO1 = np.array([0, 100, 100])
VERMELHO_ALTO1 = np.array([10, 255, 255])
VERMELHO_BAIXO2 = np.array([170, 100, 100])
VERMELHO_ALTO2 = np.array([179, 255, 255])
AREA_MINIMA = 50 # Área mínima para considerar como foco (pixeis)


# CONFIGURAÇÃO MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# FUNÇÃO PARA DETECTAR VERMELHO EM UM TILE

def detectar_areas_vermelhas_tile(imagem_rgb):
    imagem_hsv = cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2HSV)
    imagem_hsv = cv2.GaussianBlur(imagem_hsv, (5, 5), 0)

    mascara1 = cv2.inRange(imagem_hsv, VERMELHO_BAIXO1, VERMELHO_ALTO1)
    mascara2 = cv2.inRange(imagem_hsv, VERMELHO_BAIXO2, VERMELHO_ALTO2)
    mascara = cv2.bitwise_or(mascara1, mascara2)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    focos = 0
    for contorno in contornos:
        if cv2.contourArea(contorno) >= AREA_MINIMA:
            focos += 1
            cv2.drawContours(imagem_rgb, [contorno], -1, (0, 255, 0), 2)

    return focos, imagem_rgb


# FUNÇÃO PARA PROCESSAR UMA IMAGEM EM TILES

def processar_imagem_em_blocos(imagem_path):
    nome_base = os.path.basename(imagem_path).replace('.tif', '').replace('.tiff', '')

    try:
        with rasterio.open(imagem_path) as src:
            width = src.width
            height = src.height

            focos_total = 0
            imagem_final = np.zeros((height, width, 3), dtype=np.uint8)

            for y in range(0, height, TILE_SIZE):
                for x in range(0, width, TILE_SIZE):
                    janela = Window(
                        x, y,
                        min(TILE_SIZE, width - x),
                        min(TILE_SIZE, height - y)
                    )
                    bloco = src.read(window=janela)

                    if bloco.shape[0] < 3:
                        print(f"[AVISO] Menos de 3 bandas em {imagem_path}")
                        continue

                    img_rgb = np.dstack([bloco[0], bloco[1], bloco[2]]).astype(np.uint8)

                    focos_tile, imagem_processada = detectar_areas_vermelhas_tile(img_rgb)
                    focos_total += focos_tile

                    # Insere o tile processado na imagem final
                    h_tile, w_tile = imagem_processada.shape[:2]
                    imagem_final[y:y + h_tile, x:x + w_tile, :] = imagem_processada

            # Salva a imagem final com todos os contornos
            caminho_saida = os.path.join(PASTA_RESULTADOS, f"resultado_{nome_base}.png")
            cv2.imwrite(caminho_saida, cv2.cvtColor(imagem_final, cv2.COLOR_RGB2BGR))

            print(f"[RANK {rank}] Processado {nome_base} - Focos detectados: {focos_total}")

            return focos_total

    except Exception as e:
        print(f"[ERRO] Falha ao processar {imagem_path}: {e}")
        return 0


# MAIN

def main():
    if rank == 0:
        imagens = sorted([
            os.path.join(PASTA_IMAGENS, f)
            for f in os.listdir(PASTA_IMAGENS)
            if f.lower().endswith(('.tif', '.tiff'))
        ])
    else:
        imagens = None

    imagens = comm.bcast(imagens, root=0)

    imagens_local = imagens[rank::size]

    resultados_locais = [processar_imagem_em_blocos(p) for p in imagens_local]
    total_local = sum(resultados_locais)

    total_geral = comm.reduce(total_local, op=MPI.SUM, root=0)

    if rank == 0:
        print(f"\n[INFO] Total geral de areas de incendio detectadas: {total_geral}")

        fim = time.time()
        tempo_execucao = fim - inicio
        minutos = int(tempo_execucao // 60)
        segundos = int(tempo_execucao % 60)

        print(f"[INFO] Tempo total de execucao: {minutos} min {segundos} seg ({tempo_execucao:.2f} segundos)")

if __name__ == "__main__":
    main()
