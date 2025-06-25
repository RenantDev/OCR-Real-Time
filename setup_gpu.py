#!/usr/bin/env python3
"""
Script de Setup Automatizado para GPU (RTX 3060 Ti)
==================================================

Este script instala automaticamente todas as dependências necessárias
com versões específicas e compatíveis para sua RTX 3060 Ti.
"""

import subprocess
import sys
import os
import platform

def run_command(command, description):
    """Executa um comando e mostra o progresso."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Concluído!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Erro:")
        print(f"   Comando: {command}")
        print(f"   Erro: {e.stderr}")
        return False

def check_python_version():
    """Verifica se a versão do Python é compatível."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ é necessário!")
        print(f"   Versão atual: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatível!")
    return True

def check_cuda():
    """Verifica se CUDA está disponível no sistema."""
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detectada!")
            # Extrai informações da GPU
            output = result.stdout
            if "RTX 3060 Ti" in output:
                print("✅ RTX 3060 Ti detectada!")
            else:
                print("⚠️  GPU NVIDIA detectada (pode não ser RTX 3060 Ti)")
            return True
        else:
            print("❌ NVIDIA GPU não detectada!")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi não encontrado. Verifique se os drivers NVIDIA estão instalados.")
        return False

def install_dependencies():
    """Instala todas as dependências."""
    print("\n🚀 Iniciando instalação das dependências...")
    
    # Lista de comandos de instalação
    commands = [
        # Limpa cache do pip
        ("pip cache purge", "Limpando cache do pip"),
        
        # Atualiza pip
        ("python -m pip install --upgrade pip", "Atualizando pip"),
        
        # Desinstala PyTorch antigo (se existir)
        ("pip uninstall torch torchvision torchaudio -y", "Removendo PyTorch antigo"),
        
        # Instala PyTorch com CUDA 11.8
        ("pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118", "Instalando PyTorch com CUDA 11.8"),
        
        # Instala outras dependências
        ("pip install ultralytics==8.1.28", "Instalando Ultralytics YOLO"),
        ("pip install opencv-python==4.9.0.80", "Instalando OpenCV"),
        ("pip install numpy==1.24.3", "Instalando NumPy"),
        ("pip install mss==9.0.1", "Instalando MSS (captura de tela)"),
        ("pip install pyautogui==0.9.54", "Instalando PyAutoGUI"),
        ("pip install Pillow==10.2.0", "Instalando Pillow"),
        ("pip install argparse==1.4.0", "Instalando argparse"),
    ]
    
    # Executa cada comando
    for command, description in commands:
        if not run_command(command, description):
            print(f"\n❌ Falha na instalação: {description}")
            return False
    
    print("\n✅ Todas as dependências foram instaladas com sucesso!")
    return True

def test_installation():
    """Testa se a instalação foi bem-sucedida."""
    print("\n🧪 Testando instalação...")
    
    test_script = """
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import pyautogui
import mss
from PIL import Image

print("=== TESTE DE INSTALAÇÃO ===")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponível: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memória GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
else:
    print("⚠️  CUDA não disponível!")

print(f"OpenCV: {cv2.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Ultralytics: {YOLO.__version__}")
print("✅ Todas as bibliotecas importadas com sucesso!")
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", test_script], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Erro no teste:")
        print(e.stderr)
        return False

def main():
    """Função principal."""
    print("🎯 Setup Automatizado para GPU (RTX 3060 Ti)")
    print("=" * 50)
    
    # Verificações iniciais
    if not check_python_version():
        return 1
    
    if not check_cuda():
        print("\n⚠️  Continuando sem CUDA (modo CPU)...")
    
    # Instala dependências
    if not install_dependencies():
        return 1
    
    # Testa instalação
    if not test_installation():
        print("\n❌ Alguns testes falharam. Verifique as mensagens acima.")
        return 1
    
    print("\n🎉 Setup concluído com sucesso!")
    print("\n📋 Próximos passos:")
    print("   1. Execute: python main2_simple.py")
    print("   2. Execute: python segmentacao_pessoas.py")
    print("   3. Use 'q' para sair dos scripts")
    
    return 0

if __name__ == "__main__":
    exit(main()) 