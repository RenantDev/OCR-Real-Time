# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Segmentação de Pessoas em Tempo Real via Captura de Tela
=======================================================

Script para segmentação de pessoas usando YOLO11-seg.
Cria máscaras precisas da forma das pessoas detectadas.
Baseado na documentação: https://docs.ultralytics.com/pt/tasks/segment/
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import time
import torch
import pyautogui
import os
import threading
from threading import Lock

class SegmentadorPessoas:
    """Classe para segmentação de pessoas em tempo real usando YOLOv8/YOLO11-seg."""
    
    def __init__(self, modelo="yolov8n-seg.pt", confianca=0.5, tamanho_img=640, resolucao_captura=0.5, tamanho_janela=(800, 600)):
        """
        Inicializa o segmentador de pessoas.
        
        Args:
            modelo: Modelo YOLOv8/YOLO11-seg (padrão: yolov8n-seg.pt)
            confianca: Limiar de confiança (0.0 a 1.0)
            tamanho_img: Tamanho da imagem para inferência
            resolucao_captura: Fator de redução da resolução (0.1 a 1.0)
            tamanho_janela: Tamanho da janela de exibição (largura, altura)
        """
        # Verifica se CUDA está disponível
        self.device = self._verificar_gpu()
        
        # Carrega o modelo de segmentação na GPU se disponível
        print(f"🔄 Carregando modelo de segmentação: {modelo}")
        self.modelo = self._carregar_modelo(modelo)
        if self.device == "cuda":
            self.modelo.to(self.device)
            print(f"✅ GPU detectada: {torch.cuda.get_device_name()}")
            print(f"✅ Memória GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            print("⚠️  GPU não detectada - usando CPU (performance reduzida)")
        
        self.confianca = confianca
        self.tamanho_img = tamanho_img
        self.resolucao_captura = max(0.1, min(1.0, resolucao_captura))
        
        # Obtém resolução da tela
        self.largura_tela, self.altura_tela = pyautogui.size()
        self.largura_processada = int(self.largura_tela * self.resolucao_captura)
        self.altura_processada = int(self.altura_tela * self.resolucao_captura)
        
        # Calcula tamanho da janela mantendo proporção
        self.tamanho_janela = self._calcular_tamanho_janela_proporcional(tamanho_janela)
        
        self.fps_anterior = time.time()
        self.fps_atual = 0
        
        # Classe 'person' no COCO dataset é 0
        self.classe_pessoa = 0
        
        # Configura pyautogui
        pyautogui.FAILSAFE = False
        
        # Configurações de visualização
        self.mostrar_mascaras = True
        self.mostrar_contornos = True
        self.mostrar_bbox = True
        self.alpha_mascara = 0.5  # Transparência da máscara
        
        # Thread-safety lock para inferência
        self._lock = Lock()
        
        # Método de segmentação (custom ou plot nativo)
        self.usar_plot_nativo = False  # False = método custom, True = plot() nativo
        
        print(f"🎯 Modo: Segmentação de Pessoas na Tela")
        print(f"🎯 Confiança mínima: {confianca}")
        print(f"🖥️  Resolução da tela: {self.largura_tela}x{self.altura_tela}")
        print(f"🖥️  Resolução processada: {self.largura_processada}x{self.altura_processada}")
        print(f"🖥️  Fator de redução: {self.resolucao_captura:.2f}")
        print(f"🪟  Tamanho da janela: {self.tamanho_janela[0]}x{self.tamanho_janela[1]} (proporcional)")
        print(f"🎨 Visualização: Máscaras={self.mostrar_mascaras}, Contornos={self.mostrar_contornos}, BBox={self.mostrar_bbox}")
        print(f"🔒 Thread-safety: Habilitado")
        
    def _carregar_modelo(self, modelo):
        """
        Carrega o modelo YOLO, baixando automaticamente se necessário.
        
        Args:
            modelo: Nome do modelo (ex: yolo11n-seg.pt)
            
        Returns:
            Modelo YOLO carregado
        """
        try:
            # Verifica se o arquivo existe localmente
            if os.path.exists(modelo):
                print(f"📁 Modelo encontrado localmente: {modelo}")
                return YOLO(modelo)
            else:
                print(f"📥 Modelo não encontrado localmente. Baixando: {modelo}")
                print("⏳ Isso pode levar alguns minutos na primeira execução...")
                
                # Tenta baixar o modelo
                modelo_yolo = YOLO(modelo)
                print(f"✅ Modelo baixado com sucesso: {modelo}")
                return modelo_yolo
                
        except Exception as e:
            print(f"❌ Erro ao carregar modelo {modelo}: {e}")
            print("💡 Verificando modelos disponíveis...")
            
            # Lista modelos disponíveis (YOLOv8 como fallback)
            modelos_disponiveis = [
                "yolov8n-seg.pt",  # YOLOv8 nano (mais rápido)
                "yolov8s-seg.pt",  # YOLOv8 small
                "yolov8m-seg.pt",  # YOLOv8 medium
                "yolov8l-seg.pt",  # YOLOv8 large
                "yolov8x-seg.pt",  # YOLOv8 extra large (mais preciso)
                # YOLO11 quando disponível
                "yolo11n-seg.pt",
                "yolo11s-seg.pt", 
                "yolo11m-seg.pt",
                "yolo11l-seg.pt",
                "yolo11x-seg.pt"
            ]
            
            print("📋 Modelos de segmentação disponíveis:")
            for i, modelo_disp in enumerate(modelos_disponiveis, 1):
                print(f"   {i}. {modelo_disp}")
            
            # Tenta usar modelos alternativos
            modelos_fallback = ["yolov8n-seg.pt", "yolov8s-seg.pt", "yolo11n-seg.pt"]
            
            for modelo_fallback in modelos_fallback:
                if modelo_fallback != modelo:
                    try:
                        print(f"🔄 Tentando modelo alternativo: {modelo_fallback}")
                        return self._carregar_modelo(modelo_fallback)
                    except:
                        continue
            
            raise Exception("Não foi possível carregar nenhum modelo de segmentação")
        
    def _verificar_gpu(self):
        """Verifica se CUDA está disponível e retorna o dispositivo apropriado."""
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 0:
                torch.cuda.set_device(0)
                return "cuda"
        return "cpu"
        
    def _calcular_tamanho_janela_proporcional(self, tamanho_desejado):
        """
        Calcula o tamanho da janela mantendo a proporção da tela.
        
        Args:
            tamanho_desejado: (largura, altura) desejada
            
        Returns:
            (largura, altura) ajustada mantendo proporção
        """
        largura_desejada, altura_desejada = tamanho_desejado
        
        # Calcula proporção da tela
        proporcao_tela = self.largura_tela / self.altura_tela
        
        # Calcula proporção desejada
        proporcao_desejada = largura_desejada / altura_desejada
        
        if proporcao_desejada > proporcao_tela:
            # Largura é muito grande, ajusta pela altura
            nova_altura = altura_desejada
            nova_largura = int(altura_desejada * proporcao_tela)
        else:
            # Altura é muito grande, ajusta pela largura
            nova_largura = largura_desejada
            nova_altura = int(largura_desejada / proporcao_tela)
        
        return (nova_largura, nova_altura)
        
    def capturar_tela(self):
        """Captura a tela usando pyautogui e redimensiona para melhor performance."""
        try:
            # Captura screenshot
            screenshot = pyautogui.screenshot()
            
            # Converte para numpy array
            frame = np.array(screenshot)
            
            # Converte RGB para BGR (OpenCV usa BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Redimensiona para melhor performance
            frame_redimensionado = cv2.resize(
                frame, 
                (self.largura_processada, self.altura_processada), 
                interpolation=cv2.INTER_AREA
            )
            
            return frame_redimensionado
            
        except Exception as e:
            print(f"❌ Erro na captura: {e}")
            return None
        
    def calcular_fps(self):
        """Calcula e atualiza o FPS atual."""
        agora = time.time()
        self.fps_atual = 1.0 / (agora - self.fps_anterior)
        self.fps_anterior = agora
        
    def segmentar_pessoas(self, frame):
        """
        Segmenta pessoas em um frame usando YOLOv8/YOLO11-seg.
        Aplicando as melhores práticas do modo predict da documentação oficial.
        Thread-safe para uso em aplicações multithread.
        
        Args:
            frame: Frame da tela (já redimensionado)
            
        Returns:
            Frame com segmentações de pessoas desenhadas
        """
        # Thread-safety: usa lock para evitar conflitos em inferência multithread
        with self._lock:
            # Inferência de segmentação com GPU - apenas pessoas
            # Usando stream=True para melhor performance e thread-safety
            resultados = self.modelo.predict(
                frame, 
                imgsz=self.tamanho_img, 
                conf=self.confianca, 
                verbose=False,
                stream=True,  # Modo stream para melhor performance
                device=self.device,
                classes=[self.classe_pessoa],  # Filtra apenas pessoas
                iou=0.5,  # NMS IoU threshold
                max_det=20,  # Máximo de detecções
                agnostic_nms=False,  # NMS class-agnostic
                retina_masks=True,  # Máscaras de alta qualidade
                show=False,  # Não mostrar automaticamente
                save=False,  # Não salvar automaticamente
                save_txt=False,  # Não salvar labels
                save_conf=False,  # Não salvar confiança
                save_crop=False,  # Não salvar crops
                show_labels=True,  # Mostrar labels
                show_conf=True,  # Mostrar confiança
                vid_stride=1,  # Stride para vídeo
                line_width=2,  # Espessura das linhas (corrigido de line_thickness)
                visualize=False,  # Não visualizar features
                augment=False,  # Sem augmentação
                project=None,  # Projeto padrão
                name=None,  # Nome padrão
                exist_ok=False,  # Não sobrescrever
                half=False,  # Usar FP16 se disponível
                dnn=False,  # Usar OpenCV DNN
                plots=False  # Não gerar plots
            )
            
            # Processa o resultado do stream
            resultado = next(resultados)  # Pega o primeiro resultado do stream
        
        # Cria cópia do frame para desenhar
        frame_segmentado = frame.copy()
        
        # Processa cada detecção
        num_pessoas = len(resultado.boxes)
        
        if num_pessoas > 0:
            # Obtém máscaras e caixas
            if hasattr(resultado, 'masks') and resultado.masks is not None:
                mascaras = resultado.masks.data.cpu().numpy()
                caixas = resultado.boxes.xyxy.cpu().numpy()
                confiancas = resultado.boxes.conf.cpu().numpy()
                
                for i in range(num_pessoas):
                    # Obtém máscara da pessoa
                    mascara = mascaras[i]
                    
                    # Redimensiona máscara para o tamanho do frame
                    mascara_redimensionada = cv2.resize(
                        mascara.astype(np.uint8), 
                        (frame.shape[1], frame.shape[0]), 
                        interpolation=cv2.INTER_LINEAR
                    )
                    
                    # Cria máscara binária
                    mascara_binaria = (mascara_redimensionada > 0.5).astype(np.uint8)
                    
                    # Cor da máscara (verde para pessoas)
                    cor_mascara = (0, 255, 0)  # Verde
                    
                    # Aplica máscara transparente
                    if self.mostrar_mascaras:
                        frame_segmentado = self._aplicar_mascara_transparente(
                            frame_segmentado, mascara_binaria, cor_mascara, self.alpha_mascara
                        )
                    
                    # Desenha contornos da máscara
                    if self.mostrar_contornos:
                        contornos, _ = cv2.findContours(
                            mascara_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        cv2.drawContours(frame_segmentado, contornos, -1, (255, 0, 0), 2)
                    
                    # Desenha bounding box
                    if self.mostrar_bbox:
                        x1, y1, x2, y2 = map(int, caixas[i])
                        conf = confiancas[i]
                        
                        # Desenha retângulo
                        cv2.rectangle(frame_segmentado, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        
                        # Adiciona texto com confiança
                        texto = f"Pessoa: {conf:.2f}"
                        cv2.putText(frame_segmentado, texto, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame_segmentado, num_pessoas
        
    def _aplicar_mascara_transparente(self, frame, mascara, cor, alpha):
        """
        Aplica uma máscara transparente colorida sobre o frame.
        
        Args:
            frame: Frame original
            mascara: Máscara binária
            cor: Cor da máscara (B, G, R)
            alpha: Transparência (0.0 a 1.0)
            
        Returns:
            Frame com máscara aplicada
        """
        # Cria overlay colorido
        overlay = np.zeros_like(frame)
        overlay[mascara == 1] = cor
        
        # Aplica transparência
        frame_resultado = cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0)
        
        return frame_resultado
        
    def segmentar_pessoas_plot(self, frame):
        """
        Segmenta pessoas usando o método plot() da documentação oficial.
        Método alternativo mais simples usando as funcionalidades nativas do YOLO.
        
        Args:
            frame: Frame da tela (já redimensionado)
            
        Returns:
            Frame com segmentações usando plot() nativo
        """
        # Thread-safety: usa lock para evitar conflitos em inferência multithread
        with self._lock:
            # Inferência usando método plot() da documentação oficial
            resultados = self.modelo.predict(
                frame,
                imgsz=self.tamanho_img,
                conf=self.confianca,
                verbose=False,
                stream=True,
                device=self.device,
                classes=[self.classe_pessoa],
                retina_masks=True,
                show=False,
                save=False
            )
            
            # Usa o método plot() nativo do YOLO
            resultado = next(resultados)
            frame_plotado = resultado.plot(
                labels=True,      # Mostrar labels
                boxes=True,       # Mostrar bounding boxes
                masks=True,       # Mostrar máscaras
                conf=True,        # Mostrar confiança
                line_width=2      # Espessura das linhas (corrigido)
            )
            
            num_pessoas = len(resultado.boxes)
            return frame_plotado, num_pessoas
        
    def executar(self):
        """Executa o loop principal de segmentação."""
        print("\n🎮 Controles:")
        print("   Q - Sair")
        print("   M - Toggle máscaras")
        print("   C - Toggle contornos")
        print("   B - Toggle bounding boxes")
        print("   +/- - Ajustar transparência da máscara")
        print("   R - Reset configurações")
        print("   S - Salvar frame atual")
        print("   F - Toggle informações da GPU")
        print("   H - Mostrar/ocultar ajuda")
        print("   P - Toggle método de segmentação (Custom/Plot nativo)")
        print("\n🚀 Iniciando segmentação...")
        
        # Cria janela redimensionável
        cv2.namedWindow("Segmentação de Pessoas", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Segmentação de Pessoas", self.tamanho_janela[0], self.tamanho_janela[1])
        
        mostrar_ajuda = False
        mostrar_info_gpu = False
        
        try:
            while True:
                # Captura frame da tela
                frame = self.capturar_tela()
                if frame is None:
                    continue
                
                # Segmenta pessoas usando o método escolhido
                if self.usar_plot_nativo:
                    frame_segmentado, num_pessoas = self.segmentar_pessoas_plot(frame)
                else:
                    frame_segmentado, num_pessoas = self.segmentar_pessoas(frame)
                
                # Calcula FPS
                self.calcular_fps()
                
                # Adiciona informações na tela
                cv2.putText(frame_segmentado, f"FPS: {self.fps_atual:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame_segmentado, f"Pessoas: {num_pessoas}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame_segmentado, f"Confiança: {self.confianca:.2f}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame_segmentado, f"Resolução: {self.largura_processada}x{self.altura_processada}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame_segmentado, f"Método: {'Plot nativo' if self.usar_plot_nativo else 'Custom'}", 
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Mostra informações da GPU se ativado
                if mostrar_info_gpu:
                    self._mostrar_info_gpu(frame_segmentado)
                
                # Mostra ajuda se ativado
                if mostrar_ajuda:
                    self._mostrar_ajuda(frame_segmentado)
                
                # Exibe o frame
                cv2.imshow("Segmentação de Pessoas", frame_segmentado)
                
                # Processa teclas
                tecla = cv2.waitKey(1) & 0xFF
                
                if tecla == ord('q'):
                    break
                elif tecla == ord('m'):
                    self.mostrar_mascaras = not self.mostrar_mascaras
                    print(f"🎨 Máscaras: {'ON' if self.mostrar_mascaras else 'OFF'}")
                elif tecla == ord('c'):
                    self.mostrar_contornos = not self.mostrar_contornos
                    print(f"🎨 Contornos: {'ON' if self.mostrar_contornos else 'OFF'}")
                elif tecla == ord('b'):
                    self.mostrar_bbox = not self.mostrar_bbox
                    print(f"🎨 Bounding Boxes: {'ON' if self.mostrar_bbox else 'OFF'}")
                elif tecla == ord('+') or tecla == ord('='):
                    self.alpha_mascara = min(1.0, self.alpha_mascara + 0.1)
                    print(f"🎨 Transparência da máscara: {self.alpha_mascara:.1f}")
                elif tecla == ord('-'):
                    self.alpha_mascara = max(0.0, self.alpha_mascara - 0.1)
                    print(f"🎨 Transparência da máscara: {self.alpha_mascara:.1f}")
                elif tecla == ord('r'):
                    self.mostrar_mascaras = True
                    self.mostrar_contornos = True
                    self.mostrar_bbox = True
                    self.alpha_mascara = 0.5
                    print("🔄 Configurações resetadas")
                elif tecla == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    nome_arquivo = f"segmentacao_pessoas_{timestamp}.jpg"
                    cv2.imwrite(nome_arquivo, frame_segmentado)
                    print(f"💾 Frame salvo: {nome_arquivo}")
                elif tecla == ord('f'):
                    mostrar_info_gpu = not mostrar_info_gpu
                    print(f"📊 Info GPU: {'ON' if mostrar_info_gpu else 'OFF'}")
                elif tecla == ord('h'):
                    mostrar_ajuda = not mostrar_ajuda
                    print(f"❓ Ajuda: {'ON' if mostrar_ajuda else 'OFF'}")
                elif tecla == ord('p'):
                    self.usar_plot_nativo = not self.usar_plot_nativo
                    print(f"🎨 Método de segmentação: {'Custom' if not self.usar_plot_nativo else 'Plot nativo'}")
                    
        except KeyboardInterrupt:
            print("\n⏹️  Interrompido pelo usuário")
        finally:
            self.liberar()
            
    def _mostrar_info_gpu(self, frame):
        """Mostra informações da GPU no frame."""
        if self.device == "cuda":
            memoria_usada = torch.cuda.memory_allocated(0) / 1024**3
            memoria_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            utilizacao = (memoria_usada / memoria_total) * 100
            
            cv2.putText(frame, f"GPU: {torch.cuda.get_device_name()}", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Memória: {memoria_usada:.1f}GB / {memoria_total:.1f}GB", 
                       (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Utilização: {utilizacao:.1f}%", 
                       (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "GPU: CPU", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
    def _mostrar_ajuda(self, frame):
        """Mostra ajuda na tela."""
        ajuda = [
            "Q - Sair",
            "M - Toggle máscaras",
            "C - Toggle contornos", 
            "B - Toggle bounding boxes",
            "+/- - Ajustar transparência",
            "R - Reset configurações",
            "S - Salvar frame",
            "F - Info GPU",
            "H - Toggle ajuda",
            "P - Toggle método de segmentação"
        ]
        
        y_offset = 250
        for linha in ajuda:
            cv2.putText(frame, linha, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            
    def liberar(self):
        """Libera recursos."""
        cv2.destroyAllWindows()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        print("🧹 Recursos liberados")

def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description="Segmentação de pessoas em tempo real")
    parser.add_argument("--modelo", default="yolov8n-seg.pt", 
                       help="Modelo YOLOv8/YOLO11-seg (padrão: yolov8n-seg.pt)")
    parser.add_argument("--confianca", type=float, default=0.5,
                       help="Limiar de confiança (0.0 a 1.0, padrão: 0.5)")
    parser.add_argument("--tamanho", type=int, default=640,
                       help="Tamanho da imagem para inferência (padrão: 640)")
    parser.add_argument("--resolucao", type=float, default=0.5,
                       help="Fator de redução da resolução (0.1 a 1.0, padrão: 0.5)")
    parser.add_argument("--janela", nargs=2, type=int, default=[800, 600],
                       help="Tamanho da janela largura altura (padrão: 800 600)")
    
    args = parser.parse_args()
    
    # Valida argumentos
    if not (0.0 <= args.confianca <= 1.0):
        print("❌ Erro: Confiança deve estar entre 0.0 e 1.0")
        return
        
    if not (0.1 <= args.resolucao <= 1.0):
        print("❌ Erro: Resolução deve estar entre 0.1 e 1.0")
        return
    
    try:
        # Cria e executa segmentador
        segmentador = SegmentadorPessoas(
            modelo=args.modelo,
            confianca=args.confianca,
            tamanho_img=args.tamanho,
            resolucao_captura=args.resolucao,
            tamanho_janela=tuple(args.janela)
        )
        
        segmentador.executar()
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        print("💡 Dica: Certifique-se de que o modelo YOLO11-seg está disponível")
        print("💡 Execute: pip install ultralytics")

if __name__ == "__main__":
    main() 