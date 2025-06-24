# 🚀 Detector de Objetos em Tempo Real com YOLOv8

Detector de objetos otimizado para **máxima performance** usando **GPU (CUDA)**. Detecta objetos em tempo real via webcam sem lentidão.

## ✨ Características

- 🎯 **Detecção em tempo real** com YOLOv8
- 🚀 **Aceleração GPU** (CUDA) para máxima performance
- 📊 **Monitoramento de FPS** em tempo real
- 🎮 **Controles interativos** (salvar frames, info GPU)
- 🔧 **Configuração automática** de GPU
- 📱 **Interface limpa** e otimizada

## 🛠️ Instalação

### 1. Setup Automático (Recomendado)

```bash
# Execute o script de setup automático
python setup_gpu.py
```

O script irá:
- ✅ Verificar se você tem GPU NVIDIA
- ✅ Instalar PyTorch com suporte CUDA
- ✅ Configurar todas as dependências
- ✅ Testar o detector

### 2. Instalação Manual

```bash
# Instalar PyTorch com CUDA (escolha sua versão)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Instalar outras dependências
pip install ultralytics opencv-python numpy pillow matplotlib seaborn
```

## 🎮 Como Usar

### Uso Básico
```bash
python main.py
```

### Configurações Avançadas
```bash
# Alta resolução e FPS
python main.py --largura 1280 --altura 720 --fps 60

# Modelo maior para melhor precisão
python main.py --modelo yolov8s.pt --tamanho 960

# Ajustar sensibilidade
python main.py --confianca 0.7

# Usar câmera específica
python main.py --camera 1

# Forçar CPU (não recomendado)
python main.py --cpu
```

## 🎯 Controles

| Tecla | Ação |
|-------|------|
| `q` | Sair |
| `s` | Salvar frame atual |
| `i` | Mostrar informações da GPU |

## 📊 Parâmetros

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--modelo` | `yolov8n.pt` | Modelo YOLO (nano, small, medium, large) |
| `--confianca` | `0.5` | Limiar de confiança (0.0-1.0) |
| `--tamanho` | `640` | Tamanho da imagem para inferência |
| `--camera` | `0` | Índice da câmera |
| `--largura` | `640` | Largura do frame |
| `--altura` | `480` | Altura do frame |
| `--fps` | `30` | FPS desejado |
| `--cpu` | - | Forçar uso da CPU |

## 🚀 Modelos Disponíveis

| Modelo | Tamanho | Velocidade | Precisão |
|--------|---------|------------|----------|
| `yolov8n.pt` | 6.7MB | ⚡⚡⚡ | ⭐⭐ |
| `yolov8s.pt` | 22.6MB | ⚡⚡ | ⭐⭐⭐ |
| `yolov8m.pt` | 52.2MB | ⚡ | ⭐⭐⭐⭐ |
| `yolov8l.pt` | 87.7MB | 🐌 | ⭐⭐⭐⭐⭐ |

## 🔧 Requisitos do Sistema

### Mínimos
- Python 3.8+
- 4GB RAM
- Webcam

### Recomendados (para GPU)
- GPU NVIDIA com 4GB+ VRAM
- CUDA 11.8 ou 12.1
- 8GB+ RAM
- SSD para melhor performance

## 🐛 Solução de Problemas

### GPU não detectada
```bash
# Verificar drivers NVIDIA
nvidia-smi

# Verificar CUDA
nvcc --version

# Reinstalar PyTorch CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Performance baixa
- ✅ Use GPU (verifique se `Dispositivo: GPU` aparece na tela)
- ✅ Reduza resolução: `--largura 640 --altura 480`
- ✅ Use modelo menor: `--modelo yolov8n.pt`
- ✅ Ajuste FPS: `--fps 30`

### Erro de câmera
```bash
# Testar câmeras disponíveis
python main.py --camera 0
python main.py --camera 1
python main.py --camera 2
```

## 📈 Performance Esperada

| Hardware | FPS Esperado | Resolução |
|----------|--------------|-----------|
| RTX 4090 | 60+ FPS | 1920x1080 |
| RTX 3080 | 45+ FPS | 1920x1080 |
| RTX 3060 | 30+ FPS | 1280x720 |
| GTX 1660 | 25+ FPS | 1280x720 |
| CPU i7 | 5-10 FPS | 640x480 |

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

---

**🎉 Agora você tem um detector de objetos em tempo real otimizado para GPU!**
