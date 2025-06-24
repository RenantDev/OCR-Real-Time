#!/usr/bin/env python3
"""
Detector de Pessoas em Tempo Real via Captura de Tela (Versão Simples)
=====================================================================

Versão simplificada para detecção de pessoas via captura de tela.
Usa GPU (CUDA) para máxima performance.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import time
import torch
import pyautogui

class DetectorPessoasTelaSimples:
    """Classe otimizada para detecção de pessoas via captura de tela usando GPU."""
    
    def __init__(self, modelo="yolov8n.pt", confianca=0.5, tamanho_img=640, resolucao_captura=0.5, tamanho_janela=(800, 600)):
        """
        Inicializa o detector de pessoas.
        
        Args:
            modelo: Caminho para o modelo YOLO
            confianca: Limiar de confiança (0.0 a 1.0)
            tamanho_img: Tamanho da imagem para inferência
            resolucao_captura: Fator de redução da resolução (0.1 a 1.0)
            tamanho_janela: Tamanho da janela de exibição (largura, altura)
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
        self.resolucao_captura = max(0.1, min(1.0, resolucao_captura))  # Limita entre 0.1 e 1.0
        
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
        
        print(f"🎯 Modo: Detecção de Pessoas na Tela (Simples)")
        print(f"🎯 Confiança mínima: {confianca}")
        print(f"🖥️  Resolução da tela: {self.largura_tela}x{self.altura_tela}")
        print(f"🖥️  Resolução processada: {self.largura_processada}x{self.altura_processada}")
        print(f"🖥️  Fator de redução: {self.resolucao_captura:.2f}")
        print(f"🪟  Tamanho da janela: {self.tamanho_janela[0]}x{self.tamanho_janela[1]} (proporcional)")
        
    def _verificar_gpu(self):
        """Verifica se CUDA está disponível e retorna o dispositivo apropriado."""
        if torch.cuda.is_available():
            # Verifica se há GPUs disponíveis
            if torch.cuda.device_count() > 0:
                # Configura para usar a primeira GPU
                torch.cuda.set_device(0)
                return "cuda"
        return "cpu"
        
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
                interpolation=cv2.INTER_AREA  # Melhor para redução
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
        
    def detectar_pessoas(self, frame):
        """
        Detecta pessoas em um frame usando GPU.
        
        Args:
            frame: Frame da tela (já redimensionado)
            
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
        
        # Redimensiona para tamanho da janela fixo
        frame_janela = cv2.resize(
            frame_detectado, 
            self.tamanho_janela, 
            interpolation=cv2.INTER_LINEAR
        )
        
        # Adiciona informações de FPS
        self.calcular_fps()
        
        # Cor baseada no FPS (verde = bom, amarelo = médio, vermelho = ruim)
        if self.fps_atual >= 20:
            cor_fps = (0, 255, 0)  # Verde
        elif self.fps_atual >= 10:
            cor_fps = (0, 255, 255)  # Amarelo
        else:
            cor_fps = (0, 0, 255)  # Vermelho
            
        cv2.putText(
            frame_janela, 
            f"FPS: {self.fps_atual:.1f}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            cor_fps, 
            2
        )
        
        # Adiciona contador de pessoas detectadas
        num_pessoas = len(resultados.boxes)
        cv2.putText(
            frame_janela, 
            f"Pessoas: {num_pessoas}", 
            (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 255), 
            2
        )
        
        # Adiciona informação do dispositivo usado
        dispositivo_texto = "GPU" if self.device == "cuda" else "CPU"
        cv2.putText(
            frame_janela, 
            f"Dispositivo: {dispositivo_texto}", 
            (10, 90), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 255, 255), 
            2
        )
        
        # Adiciona informações de confiança média se pessoas detectadas
        if num_pessoas > 0:
            confiancas = [box.conf.item() for box in resultados.boxes]
            conf_media = sum(confiancas) / len(confiancas)
            cv2.putText(
                frame_janela, 
                f"Conf. Média: {conf_media:.2f}", 
                (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 0), 
                2
            )
            
        # Adiciona informação de resolução processada
        cv2.putText(
            frame_janela, 
            f"Processado: {self.largura_processada}x{self.altura_processada}", 
            (10, 150), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 255, 255), 
            1
        )
        
        # Adiciona informação de captura de tela
        cv2.putText(
            frame_janela, 
            "CAPTURA DE TELA - OTIMIZADA", 
            (10, 170), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 255, 255), 
            1
        )
        
        return frame_janela
    
    def executar(self):
        """Loop principal de detecção de pessoas na tela."""
        print("🚀 Iniciando detecção de pessoas na tela...")
        print("📋 Controles:")
        print("   'q' - Sair")
        print("   's' - Salvar frame atual")
        print("   'i' - Informações da GPU")
        print("   '+' - Aumentar confiança")
        print("   '-' - Diminuir confiança")
        print("   'r' - Aumentar resolução")
        print("   'f' - Diminuir resolução")
        print("   'w' - Aumentar janela")
        print("   'e' - Diminuir janela")
        
        # Cria janela redimensionável
        cv2.namedWindow("Detector de Pessoas - Tela Otimizada (GPU)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detector de Pessoas - Tela Otimizada (GPU)", self.tamanho_janela[0], self.tamanho_janela[1])
        
        try:
            while True:
                # Captura frame
                frame = self.capturar_tela()
                if frame is None:
                    print("❌ Erro ao capturar tela")
                    time.sleep(0.1)
                    continue
                
                # Detecta pessoas
                frame_processado = self.detectar_pessoas(frame)
                
                # Exibe resultado
                cv2.imshow("Detector de Pessoas - Tela Otimizada (GPU)", frame_processado)
                
                # Controle de teclas
                tecla = cv2.waitKey(1) & 0xFF
                if tecla == ord('q'):
                    break
                elif tecla == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"tela_otimizada_{timestamp}.jpg", frame_processado)
                    print(f"💾 Frame salvo: tela_otimizada_{timestamp}.jpg")
                elif tecla == ord('i'):
                    self._mostrar_info_gpu()
                elif tecla == ord('+') or tecla == ord('='):
                    self.confianca = min(1.0, self.confianca + 0.05)
                    print(f"🎯 Confiança aumentada: {self.confianca:.2f}")
                elif tecla == ord('-') or tecla == ord('_'):
                    self.confianca = max(0.1, self.confianca - 0.05)
                    print(f"🎯 Confiança diminuída: {self.confianca:.2f}")
                elif tecla == ord('r'):
                    # Aumenta resolução (diminui fator de redução)
                    self.resolucao_captura = min(1.0, self.resolucao_captura + 0.1)
                    self.largura_processada = int(self.largura_tela * self.resolucao_captura)
                    self.altura_processada = int(self.altura_tela * self.resolucao_captura)
                    print(f"🖥️  Resolução aumentada: {self.largura_processada}x{self.altura_processada} ({self.resolucao_captura:.2f})")
                elif tecla == ord('f'):
                    # Diminui resolução (aumenta fator de redução)
                    self.resolucao_captura = max(0.1, self.resolucao_captura - 0.1)
                    self.largura_processada = int(self.largura_tela * self.resolucao_captura)
                    self.altura_processada = int(self.altura_tela * self.resolucao_captura)
                    print(f"🖥️  Resolução diminuída: {self.largura_processada}x{self.altura_processada} ({self.resolucao_captura:.2f})")
                elif tecla == ord('w'):
                    # Aumenta tamanho da janela mantendo proporção
                    nova_largura = min(1920, self.tamanho_janela[0] + 100)
                    nova_altura = int(nova_largura / (self.largura_tela / self.altura_tela))
                    nova_altura = min(1080, nova_altura)
                    self.tamanho_janela = (nova_largura, nova_altura)
                    cv2.resizeWindow("Detector de Pessoas - Tela Otimizada (GPU)", self.tamanho_janela[0], self.tamanho_janela[1])
                    print(f"🪟  Janela aumentada: {self.tamanho_janela[0]}x{self.tamanho_janela[1]} (proporcional)")
                elif tecla == ord('e'):
                    # Diminui tamanho da janela mantendo proporção
                    nova_largura = max(400, self.tamanho_janela[0] - 100)
                    nova_altura = int(nova_largura / (self.largura_tela / self.altura_tela))
                    nova_altura = max(300, nova_altura)
                    self.tamanho_janela = (nova_largura, nova_altura)
                    cv2.resizeWindow("Detector de Pessoas - Tela Otimizada (GPU)", self.tamanho_janela[0], self.tamanho_janela[1])
                    print(f"🪟  Janela diminuída: {self.tamanho_janela[0]}x{self.tamanho_janela[1]} (proporcional)")
                    
        except KeyboardInterrupt:
            print("\n⏹️  Interrompido pelo usuário")
        finally:
            self.liberar()
    
    def _mostrar_info_gpu(self):
        """Mostra informações detalhadas da GPU."""
        if self.device == "cuda":
            print("\n Informações da GPU:")
            print(f"   Nome: {torch.cuda.get_device_name()}")
            print(f"   Memória Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            print(f"   Memória Alocada: {torch.cuda.memory_allocated(0) / 1024**3:.1f}GB")
            print(f"   Memória Cache: {torch.cuda.memory_reserved(0) / 1024**3:.1f}GB")
            print(f"   FPS Atual: {self.fps_atual:.1f}")
            print(f"   Confiança Atual: {self.confianca:.2f}")
            print(f"   Resolução Processada: {self.largura_processada}x{self.altura_processada}")
            print(f"   Fator de Redução: {self.resolucao_captura:.2f}")
            print(f"   Tamanho da Janela: {self.tamanho_janela[0]}x{self.tamanho_janela[1]}")
        else:
            print("\n⚠️  GPU não disponível - usando CPU")
    
    def liberar(self):
        """Libera recursos da GPU."""
        # Limpa cache da GPU
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
        cv2.destroyAllWindows()
        print("🧹 Recursos liberados")

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

def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description="Detector de Pessoas em Tempo Real via Captura de Tela (Otimizado)")
    parser.add_argument("--modelo", default="yolov8n.pt", help="Modelo YOLO (padrão: yolov8n.pt)")
    parser.add_argument("--confianca", type=float, default=0.5, help="Limiar de confiança (0.0-1.0)")
    parser.add_argument("--tamanho", type=int, default=640, help="Tamanho da imagem para inferência")
    parser.add_argument("--resolucao", type=float, default=0.5, help="Fator de redução da resolução (0.1-1.0)")
    parser.add_argument("--janela-largura", type=int, default=800, help="Largura da janela de exibição")
    parser.add_argument("--janela-altura", type=int, default=600, help="Altura da janela de exibição")
    parser.add_argument("--cpu", action="store_true", help="Forçar uso da CPU (não recomendado)")
    
    args = parser.parse_args()
    
    try:
        # Cria e configura o detector
        detector = DetectorPessoasTelaSimples(
            modelo=args.modelo,
            confianca=args.confianca,
            tamanho_img=args.tamanho,
            resolucao_captura=args.resolucao,
            tamanho_janela=(args.janela_largura, args.janela_altura)
        )
        
        # Força CPU se solicitado
        if args.cpu:
            detector.device = "cpu"
            detector.modelo.to("cpu")
            print("⚠️  Modo CPU forçado pelo usuário")
        
        # Executa a detecção
        detector.executar()
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 