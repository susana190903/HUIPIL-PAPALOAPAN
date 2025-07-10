import pygame
import cv2
import sys
import subprocess

# Inicializar pygame
pygame.init()
pygame.mixer.init()

# Cargar y reproducir el audio separado
pygame.mixer.music.load("videos/intro_audio.mp3")
pygame.mixer.music.play()

# Tamaño de la ventana
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Intro Video - Huipiles")

# Cargar el video con OpenCV
video_path = "videos/intro.mp4"
cap = cv2.VideoCapture(video_path)

# Obtener FPS del video
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps) if fps > 0 else 33

if not cap.isOpened():
    print("Error: No se pudo cargar el video")
    pygame.quit()
    sys.exit()

# Reproducir video frame por frame
clock = pygame.time.Clock()
running = True
while running:
    ret, frame = cap.read()
    if not ret:
        break  # El video ha terminado

    # Convertir BGR a RGB y redimensionar
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))

    # Convertir a Surface y mostrar en Pygame
    frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    screen.blit(frame_surface, (0, 0))
    pygame.display.update()
    clock.tick(fps)

    # Eventos
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

# Terminar video y pygame
cap.release()
pygame.quit()

# 👉 Aquí se lanza el juego "3.py"
subprocess.run([sys.executable, "3.py"])
