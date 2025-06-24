#BUENO DELAY
import subprocess # Librería para ejecutar comandos del sistema
import sys  # Asegúrate de importar esto al inicio
import pygame #Libreria para manejar gráficos, sonido y dibujar gráficos en la ventana
import time #Libreria para controlar el tiempo del juego, medir tiempos de respuesta y sincronizar eventos.
import pymunk # Librería para las colisiones de los objetos físicos
import pymunk.pygame_util # Integración de pymunk con Pygame para visualización de los objetos físicos
import json # Para cargar configuraciones desde un archivo JSON
import mediapipe as mp  # Para el reconocimiento de gestos de manos y análisis de video
import cv2  # OpenCV, utilizado para procesamiento de imágenes y video
import random # Librería para realizar operaciones aleatorias(mezclar la demostracion de las piramides)
from math import sqrt # Función matemática para calcular la raíz cuadrada (distancia entre puntos)
# Inicialización de Pygame
pygame.init()

# Abrir el archivo de configuraciones y cargar los datos
settings = open('configuraciones2.json') # Se abre el archivo JSON que contiene las configuraciones
data = json.load(settings) # Se cargan los datos del archivo JSON en el diccionario 'data'
settings.close() # Se cierra el archivo después de leer los datos
# Asignar valores de configuración desde el archivo JSON a variables
objectreadfile = data['mask'] # Archivo de la máscara u objeto
objectRadius = data['radius'] # Radio del objeto
objectColor = (data['rgb'][0],data['rgb'][1],data['rgb'][2],0) # Color del objeto en formato RGBA
makeoptimize = data["WINDOWS_opt"] # Optimización de ventanas, opción leída del JSON
invertPic = data["Inver"] # Indicador para invertir imagen, leída del JSON
gcap1 = (data['corner_1'][0],data['corner_1'][1]) # Coordenadas de la esquina superior izquierda
gcap2 = (data['corner_2'][0],data['corner_2'][1]) # Coordenadas de la esquina inferior derecha
relW = gcap2[0] - gcap1[0] # Ancho relativo entre las dos esquinas definidas
relH = gcap2[1] - gcap1[1] # Altura relativa entre las dos esquinas definida
# Variable para controlar la reproducción de GIF
playGIF = False
# Dimensiones de la pantalla de juego
SCREEN_WIDTH = 1300 # Ancho de la pantalla
SCREEN_HEIGHT = 850 # Alto de la pantalla


# Diccionario que contiene rutas a los archivos de audio de las pirámides
Audios_Salud = {
    "Huipil": r"sonido/oaxaca.mp3",
    
    
}

# Diccionario para almacenar las rutas de las imágenes de las pirámides
pyramid_images = {
    "Huipil": "imagenes/play.png",
    
}
# Lista para almacenar las pirámides colocadas correctamente
correct_pyramids = []

# Diccionario que define las áreas objetivo para cada pirámide (rectángulos para colisiones)
areas2 = {
    # Define un rectángulo con (ancho, alto)
    "Huipil": pygame.Rect(450, 650, 450, 130),
    
    
}
area3 = {
    # Define un rectángulo con (x, y, ancho, alto)
    "Huipil": pygame.Rect(520, 230, 100, 100),
   
    
}

imagenes_piramide = {
    "Huipil": "imagenes/play.png",
    
}

# Función para mezclar el orden de las pirámides aleatoriamente
def mezclar_piramides(areas2):
    # Obtener una lista de las claves (nombres de pirámides) de las áreas
    claves = list(areas2.keys())
    # Mezclar la lista de claves aleatoriamente
    random.shuffle(claves)
    # Crear un nuevo diccionario con el nuevo orden de pirámides
    nuevo_orden = {clave: areas2[clave] for clave in claves}
    return nuevo_orden
# Añadir esta función después de la función render_text:
def load_and_scale_pyramid_image(image_path, width=200, height=150):
    """Cargar y escalar una imagen de pirámide a las dimensiones especificadas"""
    try:
        image = pygame.image.load(image_path).convert_alpha()
        return pygame.transform.scale(image, (width, height))
    except pygame.error as e:
        print(f"Error al cargar la imagen {image_path}: {e}")
        # Crear una superficie de marcador de posición si falla la carga de la imagen
        surface = pygame.Surface((width, height))
        surface.fill((200, 200, 200))  # Color gris como marcador de posición
        return surface


# Llamada a la función
areas = mezclar_piramides(areas2)

# Imprimir el nuevo diccionario para ver el resultado
print(areas)

# Nuevas variables globales para el estado del juego
pyramid_released = False # Indica si la pirámide ha sido soltada
release_time = None # Guarda el tiempo cuando la pirámide fue soltada
waiting_for_result = False # Indica si se está esperando un resultado tras soltar la pirámide
pause_hand_detection = False
pause_start_time = None
pause_duration = 3  # 3 segundos de pausa en la detección
##################################3
piramides_correctas = 0 # Contador de pirámides correctas
total_piramides = len(areas) # Total de pirámides en el juego
current_audio = None # Audio actual que está sonando para una pirámide



# Inicializar sonidos usando pygame.mixer para efectos de sonido
pygame.mixer.init()
sonido_correcto = pygame.mixer.Sound("sonido/correcto.mp3")  # Sonido para acierto
sonido_incorrecto = pygame.mixer.Sound("sonido/incorrecto.mp3")  # Sonido para error


# Función para renderizar el texto con el tamaño de fuente y color especificado
def render_text(text, font_size, color):
    font = pygame.font.Font(None, font_size) # Se crea una fuente de Pygame
    return font.render(text, True, color) # Renderiza el texto con antialiasing

# Diccionario que almacena los textos renderizados para cada pirámide
pyramid_texts = {
    "Huipil": render_text("Huipil", 5, (255, 255, 255)),
    
   
}

# Lista de nombres de pirámides para mantener el orden de juego
pyramid_names = list(areas.keys())
current_pyramid_index = 0 # Índice de la pirámide actual

# Función para obtener la pirámide actual basándose en el índice
def get_current_pyramid():
    return pyramid_names[current_pyramid_index]

# Función para avanzar a la siguiente pirámide en el juego
def next_pyramid():
    global current_pyramid_index
    # Incrementar el índice de la pirámide actual
    current_pyramid_index = (current_pyramid_index + 1) % len(pyramid_names)# Incrementa el índice
    # Si el índice vuelve a ser 0, significa que el jugador ha completado todas las pirámides
    if current_pyramid_index == 0: # Si todas las pirámides han sido completadas
        pygame.mixer.music.pause() # Pausa la música
        return False  # Retorna False para indicar que el juego ha terminado
    else:
        sonido_correcto.play() # Reproduce sonido de acierto
        return True  # Continúa el juego con la siguiente pirámide

# Función para reiniciar el juego después de una colisión correcta
def reset_game():
    global start_time, current_audio, ballFrame

    moving_ball.body.position = (700, 100) # Restablece la posición de la pelota en movimiento
    moving_ball.body.velocity = (0, 0) # Restablece la velocidad de la pelota
    start_time = time.time() # Actualiza el tiempo de inicio
    
     # Obtener la pirámide actual y cargar su imagen
    current_pyramid = get_current_pyramid()
    ballFrame = pygame.image.load(imagenes_piramide[current_pyramid]).convert_alpha()
    ballFrame = pygame.transform.scale(ballFrame, (objectRadius * 2, objectRadius * 2))

    if current_pyramid in Audios_Salud: # Si hay audio asociado a la pirámide
        pygame.mixer.stop() # Detiene cualquier sonido actual
        pygame.mixer.music.load(Audios_Salud[current_pyramid]) # Carga el audio de la pirámide actual
        pygame.mixer.music.play(-1) # Reproduce el audio en bucle
        current_audio = current_pyramid # Actualiza el audio actual
        sonido_correcto.play()  # Reproduce sonido de acierto
        pygame.mixer.music.play(-1)
def handle_correct_collision():
    global game_over, piramides_correctas, current_pyramid_index, ballFrame

    current_pyramid = get_current_pyramid()

    if current_pyramid not in correct_pyramids:
        correct_pyramids.append(current_pyramid)

    sonido_correcto.play()
    piramides_correctas += 1

   

    
    if piramides_correctas == total_piramides:
        game_over = True
    else:
        current_pyramid_index = (current_pyramid_index + 1) % len(pyramid_names)

        nueva_piramide = pyramid_names[current_pyramid_index]
        print(current_pyramid_index)
        ballFrame = pygame.image.load(imagenes_piramide[nueva_piramide]).convert_alpha()
        ballFrame = pygame.transform.scale(ballFrame, (objectRadius * 2, objectRadius * 2))

        reset_game()
        # Configurar la ventana de Pygame con el tamaño definido
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT)) 
pygame.display.set_caption("HUIPILES - MUSEO TUXTEPEC") # Título de la ventana





# Cargar la imagen de fondo (un mapa con puntos) y ajustarla al tamaño de la pantalla
background_image = pygame.image.load('fondo/menu.png').convert() # Carga la imagen de fondo
background_image = pygame.transform.scale(background_image, (SCREEN_WIDTH, SCREEN_HEIGHT)) # Escala la imagen al tamaño de la ventana

# Inicializar el espacio de Pymunk para la simulación física
space = pymunk.Space()  # Crear un espacio para la simulación física
static_body = space.static_body # Cuerpo estático usado para objetos que no se mueven
# Configurar las opciones de dibujo de Pymunk utilizando Pygame
draw_options = pymunk.pygame_util.DrawOptions(screen)  # Opciones de dibujo para Pygame
# Ajustar los colores de las opciones de dibujo para evitar que se dibujen las colisiones y restricciones
draw_options.collision_point_color = (0,0,0,0)  # Desactiva el color de los puntos de colisión (transparente)
draw_options.constraint_color = (0,0,0,0) # Desactiva el color de las restricciones (transparente)

# Definir los bordes del área de simulación con líneas estáticas
lines = [
    [(0, 0), (0, SCREEN_HEIGHT)], # Línea desde la esquina superior izquierda hacia abajo
    [(0, SCREEN_HEIGHT), (SCREEN_WIDTH, SCREEN_HEIGHT)], # Línea desde la parte inferior izquierda a la derecha
    [(SCREEN_WIDTH, SCREEN_HEIGHT), (SCREEN_WIDTH, 0)], # Línea desde la parte inferior derecha hacia arriba
    [(SCREEN_WIDTH, 0), (0, 0)] # Línea desde la esquina superior derecha hacia la izquierda
]


def create_line(p1, p2, wd):
    # Crear un cuerpo estático para la simulación de física en pymunk
    # Los cuerpos estáticos no se ven afectados por las fuerzas (gravedad, colisiones, etc.)
    body = pymunk.Body(body_type=pymunk.Body.STATIC)

    # Establecer la posición inicial del cuerpo estático (en este caso, en el origen (0, 0))
    body.position = (0, 0)

    # Crear una forma de segmento (línea) utilizando los puntos de inicio y fin (p1 y p2) y el ancho (wd)
    shape = pymunk.Segment(body, p1, p2, wd)

    # Establecer la elasticidad del segmento a 0.8 (rebote moderado en colisiones)
    shape.elasticity = 0.8

    # Añadir el cuerpo y la forma al espacio de simulación de pymunk
    space.add(body, shape)


def create_ball(radius, pos, rgba):
    # Crear un cuerpo para la simulación de física en pymunk
    body = pymunk.Body()

    # Establecer la posición inicial del cuerpo
    body.position = pos

    # Crear una forma circular utilizando el cuerpo y el radio proporcionado
    shape = pymunk.Circle(body, radius)

    # Establecer la masa de la bola a 5 unidades
    shape.mass = 5

    # Establecer la elasticidad de la bola a 1 (choque completamente elástico)
    shape.elasticity = 1

    # Comentar la fricción de la bola (actualmente no se usa)
    # shape.friction = 50

    # Asignar un color RGBA a la bola para fines de visualización
    shape.color = rgba

    # Usar un pivote para añadir fricción al cuerpo
    pivot = pymunk.PivotJoint(static_body, body, (0, 0), (0, 0))

    # Deshabilitar la corrección de la articulación del pivote
    pivot.max_bias = 0

    # Emular la fricción lineal estableciendo una fuerza máxima en la articulación del pivote
    pivot.max_force = 1000

    # Añadir el cuerpo, la forma y el pivote al espacio de simulación de pymunk
    space.add(body, shape, pivot)

    # Devolver la forma creada para su uso posterior
    return shape


def create_hand_circle(position, radius=30, color=(255, 255, 255, 255)):
    # Crear un cuerpo para la simulación de física en pymunk
    body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    body.position = position

    # Crear una forma circular
    shape = pymunk.Circle(body, radius)
    shape.color = color  # Color del círculo (RGBA)

    # Añadir el cuerpo y la forma al espacio de simulación de pymunk
    space.add(body, shape)

    return body, shape

# Crear líneas físicas en el espacio
for c in lines:
    create_line(c[0],c[1],0.0)
# Inicializar variables de las pirámides y la bola en movimiento
handsShapes = [None, None]
#en esta parte se cambia la pocision en la cual se desea que se inicialice la primera piramide 
moving_ball = create_ball(round(objectRadius*1.4),(500,100),objectColor)  # Inicializar la primera pirámide en una posición inicial
frametick = 0
frameCount = 0

# Obtener el primer nombre en el diccionario 'areas'
primera_piramide = list(areas.keys())[0]
print(primera_piramide)
# Buscar la dirección en 'imagenes_piramide'
direccion_imagen = imagenes_piramide.get(primera_piramide, "Imagen no encontrada")
# Cargar la imagen de la pirámide con transparencia
objectreadfile = direccion_imagen
# Cargar la imagen de la pirámide con transparencia
pic = pygame.image.load(objectreadfile).convert_alpha()
pic = pygame.transform.scale(pic, (objectRadius*2, objectRadius*2))
ballFrame = pic # Asignar la imagen de la pirámide a ballFrame



#clock
clock = pygame.time.Clock()
FPS = 30
#colours
BG = (0, 0, 0)
run = True

makefullscreen = True

print("Ejecutando...")


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
# Configuración de la captura de video (cámara)
if makeoptimize:
    cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW) # Optimización para Windows
else:
    cap = cv2.VideoCapture(0) # Capturar video
#hola
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1300) # Establecer ancho de video
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 850) # Establecer altura de video

# Inicializar variables para el control de gestos
counter = 0
lastgestureX = 0
lastgestureY = 0
lastgestureZ = 0
moveDelta = 30
lastmoveX = 0
lastmoveY = 0
lastmoveZ = 0
waitframe = True
moveX = 0
moveY = 0
moveZ = 0
newZ = True
refZ = 0
absZ = 0
initialpose = True
zoomcounter = 0

# Función para calcular la distancia entre dos puntos
def calc_distance(p1, p2):
    return sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

# Inicializar el temporizador
start_time = time.time()
time_limit = 200  # 100 segundos para cada pirámide
game_over = False

# Agregar estas variables globales después de la definición de game_over
victory_start_time = None
victory_phase = 0  # 0: mostrar vertical, 1: mostrar horizontal, 2: mostrar mensaje final

# Iniciar el módulo de detección de manos de MediaPipe con confianza mínima de detección y seguimiento
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened() and run and not game_over:
        prueba = True
        clock.tick(FPS) # Control de velocidad de fotogramas
        space.step(10 / FPS) # Actualizar el espacio de física de Pymunk

        # Capturar un fotograma de la cámara
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frameWidth = image.shape[1]
        frameHeight = image.shape[0]

        if invertPic:
            image = cv2.flip(image, 1) # Invertir imagen si es necesario

        image.flags.writeable = False # Optimizar imagen
        results = hands.process(image) # Procesar la imagen para detectar manos
        image.flags.writeable = True # Volver a hacer la imagen escribible
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Convertir imagen de nuevo a BGR
        #se cambia el color del marco de la area de interaccion
        cv2.rectangle(image, gcap1, gcap2, (255, 255, 0), 1) 
        totalHands = 0

        # Inicializar variables para el control de la pirámide
        pyramid_held = False
        pyramid_released = False

        # Verificar si la detección de manos está pausada
        if pause_hand_detection:
            # Comprobar si el tiempo de pausa ha finalizado
            if time.time() - pause_start_time >= pause_duration:
                pause_hand_detection = False  # Reanudar la detección
                print("Detección de manos reanudada")

        # Solo procesar los resultados de las manos si la detección no está pausada
        if not pause_hand_detection and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Obtener las posiciones de los dedos relevantes
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                # Convertir las coordenadas normalizadas a píxeles
                thumb_pixel = mp_drawing._normalized_to_pixel_coordinates(thumb_tip.x, thumb_tip.y, frameWidth, frameHeight)
                index_pixel = mp_drawing._normalized_to_pixel_coordinates(index_tip.x, index_tip.y, frameWidth, frameHeight)
                
                if thumb_pixel and index_pixel:
                    # Calcular la distancia entre el pulgar e índice
                    thumb_index_distance = calc_distance(thumb_pixel, index_pixel)
                    
                    # Calcular la distancia entre el índice y el dedo medio
                    #index_middle_distance = calc_distance(index_pixel, middle_pixel)

                    # Si el pulgar y el índice están cerca, mover la pirámide
                    if thumb_index_distance < 50:
                        pyramid_held = True
                        moving_ball.body.position = index_pixel  # Mover la pirámide a la posición del índice
                    elif prueba:
                        if not pyramid_released:
                            pyramid_released = True
                            # Registrar el tiempo de liberación
                            release_time = time.time()
                            # Esperar resultado de colisión
                            waiting_for_result = True
                            # Pausar la detección de manos mientras se evalúa la colocación
                            pause_hand_detection = True
                            pause_start_time = time.time()
                            print("Detección de manos pausada por", pause_duration, "segundos")


        
        # Dibujar la pantalla de fondo y la pirámide
        screen.blit(background_image, (0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False # Salir si se cierra la ventana

        space.debug_draw(draw_options) # Dibujar objetos fisicos

        if frameCount > 15:
            frameCount = 0 # Reinicia el conteo de fotogramas cada 15 ciclos para optimizar rendimiento.

       
        # Dibujar la pirámide en su posición actual
        pyramid_pos = (int(moving_ball.body.position[0] - objectRadius), 
               int(moving_ball.body.position[1] - objectRadius))
        # Calcula la posición donde se debe dibujar la pirámide restando su radio a la posición de la bola (centrado).
        screen.blit(ballFrame, pyramid_pos) # Dibuja la pirámide en pantalla.

        # Obtener la pirámide actual y su texto asociado
        current_pyramid = get_current_pyramid() # Función que determina qué pirámide se está moviendo.
        pyramid_text = pyramid_texts[current_pyramid] # Obtiene el texto correspondiente a la pirámide actual.

        # Reproducir audio si la pirámide está en la posición correcta
        if current_audio != current_pyramid:
            pygame.mixer.music.stop() # Detiene cualquier audio previo.
            pygame.mixer.music.load(Audios_Salud[current_pyramid])  # Carga el nuevo audio asociado a la pirámide.
            pygame.mixer.music.play(-1) # Reproduce el nuevo audio en bucle.
            current_audio = current_pyramid # Actualiza la pirámide actual que está sonando.

        # Dibujar el área correcta de la pirámide
        for pyramid_name, pyramid_rect in areas.items(): # Recorre todas las áreas de pirámides.
            if pyramid_rect.collidepoint(moving_ball.body.position):  # Verifica si la pirámide actual está en su área correcta.
                pyramid_text = pyramid_texts[pyramid_name] # Actualiza el texto si la pirámide está en su lugar.
                

        text_x = moving_ball.body.position[0] - pyramid_text.get_width() // 2 # Calcula la posición del texto centrado sobre la pirámide.
        text_y = moving_ball.body.position[1] + objectRadius + 5 # Dibuja el texto justo debajo de la pirámide.
        

                # Lógica para manejar la colisión solo cuando se suelta la pirámide
        # Lógica para manejar la colisión después de esperar 5 segundos
        if waiting_for_result and time.time() - release_time >= 2:
            waiting_for_result = False
            ball_rect = pygame.Rect(moving_ball.body.position[0], moving_ball.body.position[1], 1, 1)
            correct_area = areas[current_pyramid]

            if correct_area.colliderect(ball_rect):
                print(f"¡Colisión correcta con {current_pyramid}!")
                sonido_correcto.play()
                handle_correct_collision()

                # Ejecutar 4.py y cerrar este proceso
                pygame.mixer.music.stop()
                cap.release()
                cv2.destroyAllWindows()
                pygame.quit()
                try:
                    subprocess.Popen(['python', '3.py'])
                except Exception as e:
                    print(f"Error al ejecutar 3.py: {e}")
                sys.exit()

            else:
                collision_detected = False
                for name, area in areas.items():
                    if area.colliderect(ball_rect):
                        print(f"Colisión incorrecta. La pirámide correcta es {current_pyramid}")
                        sonido_incorrecto.play()
                        game_over = True
                        collision_detected = True
                        break
                if not collision_detected:
                    print("No se detectó colisión con ninguna área.")
                    

         # Reiniciar el estado si la pirámide se mueve durante la espera
        if waiting_for_result and pyramid_held:
            waiting_for_result = False
            pyramid_released = False
            # También deberíamos cancelar la pausa si el usuario vuelve a agarrar la pirámide
            pause_hand_detection = False
        # Crear superficies transparentes para las áreas una vez
        area_surfaces = {}
        for name, area in areas.items():
            surface = pygame.Surface((area.width, area.height), pygame.SRCALPHA)
            #surface.fill((255,0,0))  # Completamente transparente
            surface.fill((0, 0, 0, 0))  # Completamente transparente
            
            pygame.draw.rect(surface, (0, 255, 0, 0), (0, 0, area.width, area.height), 2)  # Borde semi-transparente
            area_surfaces[name] = surface
        for name, area in areas.items():
            color = (255, 255, 255, 0) if name == current_pyramid else (255, 255, 255, 0)
            surface = area_surfaces[name].copy()
            pygame.draw.rect(surface, color, (0, 0, area.width, area.height), 2)  # Actualizar el color del borde
            screen.blit(surface, (area.x, area.y))



        elapsed_time = time.time() - start_time
        remaining_time = max(0, time_limit - elapsed_time)
        time_text = render_text(f"Tiempo: {int(remaining_time)}s", 15, (255, 255, 255))
        time_x = 10
        time_y = 10
        screen.blit(time_text, (time_x, time_y))
        #CAMBIO 2
        if remaining_time <= 0 or piramides_correctas == 4:
            pygame.mixer.music.pause()
            game_over = True

        if frametick > 0:
            frametick = 0
            frameCount += 1
        frametick += 1

        

        for pyramid_name in correct_pyramids:
            if pyramid_name in pyramid_images:
                # Obtener el área del libro donde se mostrará esta pirámide
                book_area = area3[pyramid_name]
                
                # Cargar y escalar la imagen de la pirámide
                image = load_and_scale_pyramid_image(pyramid_images[pyramid_name], 
                                                    width=book_area.width, 
                                                    height=book_area.height)
                
                # Dibujar la imagen en su área correspondiente en el libro
                screen.blit(image, (book_area.x, book_area.y))
                
        pygame.display.update()


    if game_over:
        #screen.fill(BG)
        screen.blit(background_image, (0, 0))
        #CAMBIO 3
        if piramides_correctas == 1:
            current_time = time.time()

            # Inicializar el tiempo de victoria si es necesario
            if victory_start_time is None:
                victory_start_time = current_time
            
            elapsed_victory_time = current_time - victory_start_time
            


            #print("1")
        else:
            # Definir el texto 'Game Over'
            # Renderizar el texto "Game Over"
            game_over_text = render_text("FIN DEL JUEGO", 100, (0,0,0))  # Texto "Game Over" con tamaño 50 y color rojo
            
            # Centrar el texto "Game Over" en pantalla
            text_x = SCREEN_WIDTH // 2 - game_over_text.get_width() // 2
            text_y = SCREEN_HEIGHT // 2 - game_over_text.get_height() // 2
            
            # Mostrar el texto "Game Over" en pantalla
            screen.blit(game_over_text, (text_x, text_y))
            
           

    pygame.display.update()
# Variables para el temporizador
start_time = time.time()
timeout = 10  # 10 segundos

# Variable para controlar el bucle del juego
waiting_for_enter = True

while waiting_for_enter:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            waiting_for_enter = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                waiting_for_enter = False
    
    # Comprobar el temporizador
    current_time = time.time()
    elapsed_time = current_time - start_time
    if elapsed_time > timeout:
        waiting_for_enter = False

    pygame.mixer.music.stop()

cap.release()
cv2.destroyAllWindows()
pygame.quit()

