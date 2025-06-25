# -*- coding: utf-8 -*-
"""
Exemplo de uso do modo predict do Ultralytics YOLO
Baseado na documentação oficial: https://docs.ultralytics.com/pt/modes/predict/

Este script demonstra as melhores práticas para inferência em tempo real
usando as funcionalidades recomendadas pela documentação oficial.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch
import threading
from threading import Lock

class ExemploPredict:
    """Exemplo de implementação seguindo as melhores práticas da documentação oficial."""
    
    def __init__(self, modelo="yolov8n-seg.pt"):
        """
        Inicializa o exemplo seguindo as recomendações da documentação.
        
        Args:
            modelo: Modelo YOLO para usar
        """
        # Verifica GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Dispositivo: {self.device}")
        
        # Carrega modelo
        print(f"📥 Carregando modelo: {modelo}")
        self.modelo = YOLO(modelo)
        if self.device == "cuda":
            self.modelo.to(self.device)
        
        # Thread-safety lock (recomendado na documentação)
        self._lock = Lock()
        
        # Configurações de inferência (baseadas na documentação)
        self.conf = 0.5
        self.iou = 0.5
        self.imgsz = 640
        self.max_det = 20
        
        print("✅ Exemplo inicializado seguindo as melhores práticas da documentação")
        
    def inferencia_basica(self, imagem):
        """
        Inferência básica seguindo a documentação oficial.
        
        Args:
            imagem: Imagem para processar
            
        Returns:
            Resultados da inferência
        """
        # Thread-safety conforme documentação
        with self._lock:
            resultados = self.modelo.predict(
                imagem,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                max_det=self.max_det,
                verbose=False,
                stream=True,  # Modo stream para melhor performance
                device=self.device
            )
            return next(resultados)
    
    def inferencia_com_plot(self, imagem):
        """
        Inferência usando método plot() da documentação.
        
        Args:
            imagem: Imagem para processar
            
        Returns:
            Imagem com anotações
        """
        with self._lock:
            resultados = self.modelo.predict(
                imagem,
                conf=self.conf,
                imgsz=self.imgsz,
                verbose=False,
                stream=True,
                device=self.device
            )
            
            resultado = next(resultados)
            
            # Usa método plot() da documentação
            imagem_plotada = resultado.plot(
                labels=True,      # Mostrar labels
                boxes=True,       # Mostrar bounding boxes
                masks=True,       # Mostrar máscaras (se disponível)
                conf=True,        # Mostrar confiança
                line_width=2      # Espessura das linhas (corrigido)
            )
            
            return imagem_plotada, len(resultado.boxes)
    
    def inferencia_multithread(self, imagens):
        """
        Exemplo de inferência multithread conforme documentação.
        
        Args:
            imagens: Lista de imagens para processar
            
        Returns:
            Lista de resultados
        """
        def processar_imagem(imagem, resultados, idx):
            """Função para processar uma imagem em thread separada."""
            # Cada thread cria sua própria instância do modelo (thread-safety)
            modelo_thread = YOLO(self.modelo.ckpt_path)
            if self.device == "cuda":
                modelo_thread.to(self.device)
            
            resultado = modelo_thread.predict(
                imagem,
                conf=self.conf,
                imgsz=self.imgsz,
                verbose=False,
                device=self.device
            )[0]
            
            resultados[idx] = resultado
        
        # Processa imagens em threads separadas
        threads = []
        resultados = [None] * len(imagens)
        
        for i, imagem in enumerate(imagens):
            thread = threading.Thread(
                target=processar_imagem,
                args=(imagem, resultados, i)
            )
            threads.append(thread)
            thread.start()
        
        # Aguarda todas as threads terminarem
        for thread in threads:
            thread.join()
        
        return resultados
    
    def exemplo_stream_video(self, caminho_video):
        """
        Exemplo de processamento de vídeo usando stream conforme documentação.
        
        Args:
            caminho_video: Caminho para o arquivo de vídeo
        """
        cap = cv2.VideoCapture(caminho_video)
        
        if not cap.isOpened():
            print(f"❌ Erro ao abrir vídeo: {caminho_video}")
            return
        
        print("🎬 Processando vídeo com stream...")
        
        while cap.isOpened():
            success, frame = cap.read()
            
            if not success:
                break
            
            # Processa frame usando stream
            with self._lock:
                resultados = self.modelo.predict(
                    frame,
                    conf=self.conf,
                    imgsz=self.imgsz,
                    verbose=False,
                    stream=True,
                    device=self.device
                )
                
                resultado = next(resultados)
                frame_processado = resultado.plot()
            
            # Exibe resultado
            cv2.imshow("Vídeo Processado", frame_processado)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Processamento de vídeo concluído")
    
    def exemplo_batch_inference(self, imagens):
        """
        Exemplo de inferência em lote conforme documentação.
        
        Args:
            imagens: Lista de imagens para processar em lote
            
        Returns:
            Lista de resultados
        """
        print(f"📦 Processando {len(imagens)} imagens em lote...")
        
        with self._lock:
            # Inferência em lote (mais eficiente)
            resultados = self.modelo.predict(
                imagens,
                conf=self.conf,
                imgsz=self.imgsz,
                verbose=False,
                device=self.device
            )
        
        print(f"✅ Processamento em lote concluído: {len(resultados)} resultados")
        return resultados

def main():
    """Função principal demonstrando os exemplos."""
    print("🚀 Exemplo de uso do modo predict do Ultralytics YOLO")
    print("📚 Baseado na documentação oficial: https://docs.ultralytics.com/pt/modes/predict/")
    print()
    
    # Cria exemplo
    exemplo = ExemploPredict("yolov8n-seg.pt")
    
    # Exemplo com webcam
    print("📹 Iniciando exemplo com webcam...")
    print("💡 Pressione 'q' para sair")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Erro ao abrir webcam")
        return
    
    try:
        while True:
            success, frame = cap.read()
            
            if not success:
                break
            
            # Processa frame usando método plot() da documentação
            frame_processado, num_deteccoes = exemplo.inferencia_com_plot(frame)
            
            # Adiciona informações
            cv2.putText(frame_processado, f"Deteções: {num_deteccoes}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_processado, "Pressione 'q' para sair", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Exibe resultado
            cv2.imshow("Exemplo Predict - Documentação Oficial", frame_processado)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n⏹️  Interrompido pelo usuário")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Exemplo concluído")

if __name__ == "__main__":
    main() 