import pygame
import cv2
import sys
import subprocess

# Inicializar pygame
pygame.init()
pygame.mixer.init()

# Obtener resolución de pantalla
screen_info = pygame.display.Info()
SCREEN_WIDTH = screen_info.current_w
SCREEN_HEIGHT = screen_info.current_h

# Poner ventana en fullscreen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Intro Video - Huipiles")

# Cargar y reproducir el audio
pygame.mixer.music.load("videos/intro_audio.mp3")
pygame.mixer.music.play()

# Cargar el video
video_path = "videos/intro.mp4"
cap = cv2.VideoCapture(video_path)

# Obtener FPS del video
fps = cap.get(cv2.CAP_PROP_FPS)
clock = pygame.time.Clock()

if not cap.isOpened():
    print("Error: No se pudo abrir el video.")
    pygame.quit()
    sys.exit()

# Reproducir video frame por frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir BGR a RGB y escalar al tamaño de la pantalla
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))

    # Mostrar en Pygame
    frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    screen.blit(frame_surface, (0, 0))
    pygame.display.update()
    clock.tick(fps)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            cap.release()
            sys.exit()

# Terminar video
cap.release()
pygame.quit()

# Abrir el juego 3.py automáticamente
subprocess.run([sys.executable, "3.py"])

