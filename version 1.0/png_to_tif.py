from PIL import Image
import os

def converter_imagens_para_tif(diretorio_origem, diretorio_destino):
    # Garante que o diretório de saída existe
    os.makedirs(diretorio_destino, exist_ok=True)

    # Definir os formatos de imagem permitidos para conversão
    formatos_permitidos = ('.png', '.jpg', '.jpeg')

    # Lista todos os arquivos no diretório de entrada
    arquivos = os.listdir(diretorio_origem)

    for arquivo in arquivos:
        if arquivo.lower().endswith(formatos_permitidos):
            caminho_entrada = os.path.join(diretorio_origem, arquivo)
            nome_arquivo = os.path.splitext(arquivo)[0]
            caminho_saida = os.path.join(diretorio_destino, f"{nome_arquivo}.tif")

            try:
                with Image.open(caminho_entrada) as imagem:
                    imagem_rgb = imagem.convert('RGB')
                    imagem_rgb.save(caminho_saida, format='TIFF')
                    print(f"[SUCESSO] {arquivo} convertido para {caminho_saida}")
            except Exception as erro:
                print(f"[FALHA] Não foi possível converter {arquivo}: {erro}")

# Executando a função
if __name__ == "__main__":
    diretorio_entrada = os.path.join(os.getcwd(), "imagens_satelite")
    diretorio_saida = os.path.join(os.getcwd(), "imagens_em_tif")
    converter_imagens_para_tif(diretorio_entrada, diretorio_saida)
