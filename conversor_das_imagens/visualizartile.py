import cv2
import tifffile as tiff
import os

def salvar_tile(imagem_path, x=0, y=0, tile_size=1024, nome_saida="tile_extraido.jpg"):
    print(f"🖼️ Abrindo imagem: {imagem_path}")

    ext = os.path.splitext(imagem_path)[1].lower()
    if ext in [".tif", ".tiff"]:
        with tiff.TiffFile(imagem_path) as tif:
            img = tif.asarray()
    else:
        img = cv2.imread(imagem_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError("❌ Imagem não encontrada ou inválida.")

    altura, largura = img.shape[:2]
    print(f"📐 Dimensões da imagem: {largura}x{altura}")

    tile = img[y:min(y+tile_size, altura), x:min(x+tile_size, largura)]

    if tile.ndim == 2:
        tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
    elif tile.shape[2] == 4:
        tile = cv2.cvtColor(tile, cv2.COLOR_BGRA2BGR)

    # ✅ Corrigir cor RGB para BGR
    tile_bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)

    # 📁 Garante que a pasta "tiles" existe
    pasta_saida = "tiles"
    os.makedirs(pasta_saida, exist_ok=True)

    caminho_saida = os.path.join(pasta_saida, nome_saida)
    cv2.imwrite(caminho_saida, tile_bgr)

    print(f"✅ Tile salvo em: {caminho_saida}")

# === CONFIGURAÇÃO ===
CAMINHO_IMAGEM = "D:/fire_detector/fire_detector/img_gigantes/imagem_gigante_20GB.tiff"
X_TILE = 0
Y_TILE = 0
TAMANHO_TILE = 1024
SAIDA = "tile_inspecao.jpg"  # nome do arquivo dentro da pasta tiles/

if __name__ == "__main__":
    salvar_tile(CAMINHO_IMAGEM, X_TILE, Y_TILE, TAMANHO_TILE, SAIDA)
