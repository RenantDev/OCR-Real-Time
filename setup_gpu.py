#!/usr/bin/env python3
"""
Script de Setup para Detector de Objetos com GPU
===============================================

Verifica e configura automaticamente o ambiente para usar GPU.
"""

import subprocess
import sys
import os

def executar_comando(comando):
    """Executa um comando e retorna o resultado."""
    try:
        resultado = subprocess.run(comando, shell=True, capture_output=True, text=True)
        return resultado.returncode == 0, resultado.stdout, resultado.stderr
    except Exception as e:
        return False, "", str(e)

def verificar_cuda():
    """Verifica se CUDA está instalado no sistema."""
    print("🔍 Verificando CUDA...")
    
    # Verifica nvidia-smi
    sucesso, saida, erro = executar_comando("nvidia-smi")
    if sucesso:
        print("✅ NVIDIA GPU detectada:")
        print(saida)
        return True
    else:
        print("❌ NVIDIA GPU não detectada ou drivers não instalados")
        print("   Erro:", erro)
        return False

def verificar_pytorch_cuda():
    """Verifica se PyTorch tem suporte CUDA."""
    print("\n🔍 Verificando PyTorch CUDA...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ PyTorch CUDA disponível")
            print(f"   Versão PyTorch: {torch.__version__}")
            print(f"   Versão CUDA: {torch.version.cuda}")
            print(f"   GPUs disponíveis: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("❌ PyTorch instalado mas sem suporte CUDA")
            return False
    except ImportError:
        print("❌ PyTorch não instalado")
        return False

def instalar_dependencias():
    """Instala as dependências necessárias."""
    print("\n📦 Instalando dependências...")
    
    # Lista de dependências
    dependencias = [
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0", 
        "numpy>=1.24.0",
        "pillow>=9.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0"
    ]
    
    for dep in dependencias:
        print(f"   Instalando {dep}...")
        sucesso, saida, erro = executar_comando(f"pip install {dep}")
        if sucesso:
            print(f"   ✅ {dep} instalado")
        else:
            print(f"   ❌ Erro ao instalar {dep}: {erro}")

def instalar_pytorch_cuda():
    """Instala PyTorch com suporte CUDA."""
    print("\n🚀 Instalando PyTorch com CUDA...")
    
    # Tenta detectar a versão do CUDA
    sucesso, saida, erro = executar_comando("nvcc --version")
    if sucesso:
        # Extrai versão do CUDA
        for linha in saida.split('\n'):
            if 'release' in linha.lower():
                versao = linha.split('release')[1].split(',')[0].strip()
                print(f"   Versão CUDA detectada: {versao}")
                
                # Instala versão apropriada do PyTorch
                if '11.8' in versao:
                    comando = "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
                elif '12.1' in versao:
                    comando = "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
                else:
                    comando = "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
                
                print(f"   Executando: {comando}")
                sucesso, saida, erro = executar_comando(comando)
                if sucesso:
                    print("   ✅ PyTorch CUDA instalado com sucesso")
                    return True
                else:
                    print(f"   ❌ Erro: {erro}")
                    return False
    else:
        print("   ⚠️  Não foi possível detectar versão do CUDA")
        print("   Tentando instalar versão padrão...")
        
        comando = "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
        sucesso, saida, erro = executar_comando(comando)
        if sucesso:
            print("   ✅ PyTorch CUDA instalado")
            return True
        else:
            print(f"   ❌ Erro: {erro}")
            return False

def testar_detector():
    """Testa se o detector funciona corretamente."""
    print("\n🧪 Testando detector...")
    
    try:
        from ultralytics import YOLO
        import torch
        
        # Testa carregamento do modelo
        modelo = YOLO("yolov8n.pt")
        
        # Testa GPU
        if torch.cuda.is_available():
            modelo.to("cuda")
            print("✅ Teste GPU: OK")
        else:
            print("⚠️  GPU não disponível - usando CPU")
        
        print("✅ Detector funcionando corretamente")
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        return False

def main():
    """Função principal."""
    print("🚀 Setup do Detector de Objetos com GPU")
    print("=" * 50)
    
    # Verifica CUDA
    cuda_disponivel = verificar_cuda()
    
    # Verifica PyTorch
    pytorch_ok = verificar_pytorch_cuda()
    
    if not pytorch_ok:
        print("\n📦 PyTorch CUDA não encontrado. Instalando...")
        if instalar_pytorch_cuda():
            pytorch_ok = True
        else:
            print("❌ Falha ao instalar PyTorch CUDA")
            return 1
    
    # Instala outras dependências
    instalar_dependencias()
    
    # Testa o detector
    if testar_detector():
        print("\n🎉 Setup concluído com sucesso!")
        print("\n📋 Para usar o detector:")
        print("   python main.py")
        print("\n📋 Para forçar CPU (não recomendado):")
        print("   python main.py --cpu")
        return 0
    else:
        print("\n❌ Setup falhou")
        return 1

if __name__ == "__main__":
    exit(main()) 