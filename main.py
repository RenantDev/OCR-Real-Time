#!/usr/bin/env python3
"""
Detector de Pessoas em Tempo Real com YOLOv8
============================================

Versão otimizada para detecção de pessoas via webcam sem lentidão.
Usa GPU (CUDA) para máxima performance.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import time
import torch

class DetectorPessoas:
    """Classe otimizada para detecção de pessoas em tempo real usando GPU."""
    
    def __init__(self, modelo="yolov8n.pt", confianca=0.5, tamanho_img=640):
        """
        Inicializa o detector de pessoas.
        
        Args:
            modelo: Caminho para o modelo YOLO
            confianca: Limiar de confiança (0.0 a 1.0)
            tamanho_img: Tamanho da imagem para inferência
        """
        # Verifica se CUDA está disponível
        self.device = self._verificar_gpu()
        
        # Carrega o modelo na GPU se disponível
        self.modelo = YOLO(modelo)
        if self.device == "cuda":
            self.modelo.to(self.device)
            print(f"✅ GPU detectada: {torch.cuda.get_device_name()}")
            print(f"✅ Memória GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            print("⚠️  GPU não detectada - usando CPU (performance reduzida)")
        
        self.confianca = confianca
        self.tamanho_img = tamanho_img
        self.fps_anterior = time.time()
        self.fps_atual = 0
        
        # Classe 'person' no COCO dataset é 0
        self.classe_pessoa = 0
        
        print(f"🎯 Modo: Detecção de Pessoas")
        print(f"🎯 Confiança mínima: {confianca}")
        
    def _verificar_gpu(self):
        """Verifica se CUDA está disponível e retorna o dispositivo apropriado."""
        if torch.cuda.is_available():
            # Verifica se há GPUs disponíveis
            if torch.cuda.device_count() > 0:
                # Configura para usar a primeira GPU
                torch.cuda.set_device(0)
                return "cuda"
        return "cpu"
        
    def configurar_camera(self, dispositivo=0, largura=640, altura=480, fps=30):
        """
        Configura a câmera com parâmetros otimizados.
        
        Args:
            dispositivo: Índice da câmera (0, 1, 2...)
            largura: Largura do frame
            altura: Altura do frame
            fps: FPS desejado
        """
        self.cap = cv2.VideoCapture(dispositivo)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir a câmera {dispositivo}")
        
        # Configurações otimizadas para performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo para reduzir latência
        
        # Configurações adicionais para melhor performance
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Desabilita autofoco
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Exposição manual
        
        print(f"📹 Câmera configurada: {largura}x{altura} @ {fps}fps")
        
    def calcular_fps(self):
        """Calcula e atualiza o FPS atual."""
        agora = time.time()
        self.fps_atual = 1.0 / (agora - self.fps_anterior)
        self.fps_anterior = agora
        
    def detectar_pessoas(self, frame):
        """
        Detecta pessoas em um frame usando GPU.
        
        Args:
            frame: Frame da câmera
            
        Returns:
            Frame com detecções de pessoas desenhadas
        """
        # Inferência otimizada com GPU - apenas pessoas
        resultados = self.modelo.predict(
            frame, 
            imgsz=self.tamanho_img, 
            conf=self.confianca, 
            verbose=False,
            stream=False,  # Desabilita streaming para melhor performance
            device=self.device,  # Força uso da GPU
            classes=[self.classe_pessoa]  # Filtra apenas pessoas
        )[0]
        
        # Desenha as detecções
        frame_detectado = resultados.plot()
        
        # Adiciona informações de FPS
        self.calcular_fps()
        
        # Cor baseada no FPS (verde = bom, amarelo = médio, vermelho = ruim)
        if self.fps_atual >= 25:
            cor_fps = (0, 255, 0)  # Verde
        elif self.fps_atual >= 15:
            cor_fps = (0, 255, 255)  # Amarelo
        else:
            cor_fps = (0, 0, 255)  # Vermelho
            
        cv2.putText(
            frame_detectado, 
            f"FPS: {self.fps_atual:.1f}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            cor_fps, 
            2
        )
        
        # Adiciona contador de pessoas detectadas
        num_pessoas = len(resultados.boxes)
        cv2.putText(
            frame_detectado, 
            f"Pessoas: {num_pessoas}", 
            (10, 70), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 255), 
            2
        )
        
        # Adiciona informação do dispositivo usado
        dispositivo_texto = "GPU" if self.device == "cuda" else "CPU"
        cv2.putText(
            frame_detectado, 
            f"Dispositivo: {dispositivo_texto}", 
            (10, 110), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        
        # Adiciona informações de confiança média se pessoas detectadas
        if num_pessoas > 0:
            confiancas = [box.conf.item() for box in resultados.boxes]
            conf_media = sum(confiancas) / len(confiancas)
            cv2.putText(
                frame_detectado, 
                f"Conf. Média: {conf_media:.2f}", 
                (10, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 0), 
                2
            )
        
        return frame_detectado
    
    def executar(self):
        """Loop principal de detecção de pessoas."""
        print("🚀 Iniciando detecção de pessoas...")
        print("📋 Controles:")
        print("   'q' - Sair")
        print("   's' - Salvar frame atual")
        print("   'i' - Informações da GPU")
        print("   '+' - Aumentar confiança")
        print("   '-' - Diminuir confiança")
        
        try:
            while True:
                # Captura frame
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Erro ao capturar frame")
                    break
                
                # Detecta pessoas
                frame_processado = self.detectar_pessoas(frame)
                
                # Exibe resultado
                cv2.imshow("Detector de Pessoas - YOLOv8 (GPU)", frame_processado)
                
                # Controle de teclas
                tecla = cv2.waitKey(1) & 0xFF
                if tecla == ord('q'):
                    break
                elif tecla == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"pessoas_{timestamp}.jpg", frame_processado)
                    print(f"💾 Frame salvo: pessoas_{timestamp}.jpg")
                elif tecla == ord('i'):
                    self._mostrar_info_gpu()
                elif tecla == ord('+') or tecla == ord('='):
                    self.confianca = min(1.0, self.confianca + 0.05)
                    print(f"🎯 Confiança aumentada: {self.confianca:.2f}")
                elif tecla == ord('-') or tecla == ord('_'):
                    self.confianca = max(0.1, self.confianca - 0.05)
                    print(f"🎯 Confiança diminuída: {self.confianca:.2f}")
                    
        except KeyboardInterrupt:
            print("\n⏹️  Interrompido pelo usuário")
        finally:
            self.liberar()
    
    def _mostrar_info_gpu(self):
        """Mostra informações detalhadas da GPU."""
        if self.device == "cuda":
            print("\n🔍 Informações da GPU:")
            print(f"   Nome: {torch.cuda.get_device_name()}")
            print(f"   Memória Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            print(f"   Memória Alocada: {torch.cuda.memory_allocated(0) / 1024**3:.1f}GB")
            print(f"   Memória Cache: {torch.cuda.memory_reserved(0) / 1024**3:.1f}GB")
            print(f"   FPS Atual: {self.fps_atual:.1f}")
            print(f"   Confiança Atual: {self.confianca:.2f}")
        else:
            print("\n⚠️  GPU não disponível - usando CPU")
    
    def liberar(self):
        """Libera recursos da câmera e GPU."""
        if hasattr(self, 'cap'):
            self.cap.release()
        
        # Limpa cache da GPU
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
        cv2.destroyAllWindows()
        print("🧹 Recursos liberados")

def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description="Detector de Pessoas em Tempo Real com GPU")
    parser.add_argument("--modelo", default="yolov8n.pt", help="Modelo YOLO (padrão: yolov8n.pt)")
    parser.add_argument("--confianca", type=float, default=0.5, help="Limiar de confiança (0.0-1.0)")
    parser.add_argument("--tamanho", type=int, default=640, help="Tamanho da imagem para inferência")
    parser.add_argument("--camera", type=int, default=0, help="Índice da câmera")
    parser.add_argument("--largura", type=int, default=640, help="Largura do frame")
    parser.add_argument("--altura", type=int, default=480, help="Altura do frame")
    parser.add_argument("--fps", type=int, default=30, help="FPS desejado")
    parser.add_argument("--cpu", action="store_true", help="Forçar uso da CPU (não recomendado)")

    args = parser.parse_args()

    try:
        # Cria e configura o detector
        detector = DetectorPessoas(
            modelo=args.modelo,
            confianca=args.confianca,
            tamanho_img=args.tamanho
        )
        
        # Força CPU se solicitado
        if args.cpu:
            detector.device = "cpu"
            detector.modelo.to("cpu")
            print("⚠️  Modo CPU forçado pelo usuário")
        
        # Configura a câmera
        detector.configurar_camera(
            dispositivo=args.camera,
            largura=args.largura,
            altura=args.altura,
            fps=args.fps
        )
        
        # Executa a detecção
        detector.executar()
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
