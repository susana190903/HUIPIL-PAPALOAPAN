# BUENO
import subprocess # Librería para ejecutar comandos del sistema
import sys  # Asegúrate de importar esto al inicio
import pygame  # Libreria para manejar gráficos, sonido y dibujar gráficos en la ventana
import time  # Libreria para controlar el tiempo del juego, medir tiempos de respuesta y sincronizar eventos.
import pymunk  # Librería para las colisiones de los objetos físicos
import pymunk.pygame_util  # Integración de pymunk con Pygame para visualización de los objetos físicos
import json  # Para cargar configuraciones desde un archivo JSON
import mediapipe as mp  # Para el reconocimiento de gestos de manos y análisis de video
import cv2  # OpenCV, utilizado para procesamiento de imágenes y video
import random  # Librería para realizar operaciones aleatorias(mezclar la demostracion de las piramides)
from math import sqrt  # Función matemática para calcular la raíz cuadrada (distancia entre puntos)



# Inicialización de Pygame
pygame.init()

# Abrir el archivo de configuraciones y cargar los datos
settings = open('configuraciones.json')  # Se abre el archivo JSON que contiene las configuraciones
data = json.load(settings)  # Se cargan los datos del archivo JSON en el diccionario 'data'
settings.close()  # Se cierra el archivo después de leer los datos
# Asignar valores de configuración desde el archivo JSON a variables
objectreadfile = data['mask']  # Archivo de la máscara u objeto
objectRadius = data['radius']  # Radio del objeto
objectColor = (data['rgb'][0], data['rgb'][1], data['rgb'][2], 0)  # Color del objeto en formato RGBA
makeoptimize = data["WINDOWS_opt"]  # Optimización de ventanas, opción leída del JSON
invertPic = data["Inver"]  # Indicador para invertir imagen, leída del JSON
gcap1 = (data['corner_1'][0], data['corner_1'][1])  # Coordenadas de la esquina superior izquierda
gcap2 = (data['corner_2'][0], data['corner_2'][1])  # Coordenadas de la esquina inferior derecha
relW = gcap2[0] - gcap1[0]  # Ancho relativo entre las dos esquinas definidas
relH = gcap2[1] - gcap1[1]  # Altura relativa entre las dos esquinas definida
# Variable para controlar la reproducción de GIF
playGIF = False
# Dimensiones de la pantalla de juego
SCREEN_WIDTH = 1300  # Ancho de la pantalla
SCREEN_HEIGHT = 850  # Alto de la pantalla
 
def render_text(text, font_size, color):
    font = pygame.font.Font(None, font_size)  # Se crea una fuente de Pygame
    return font.render(text, True, color)  # Renderiza el texto con antialiasing



# Diccionario que contiene rutas a los archivos de audio de las pirámides
audios_piramides = {
    "San Felipe Usila": r"Audios_Piramides/Chichén_Itzá.mp3",
    "San Miguel Soyaltepec": r"Audios_Piramides/Chichén_Itzá.mp3",
    "San Juan Bautista Tuxtepec": r"Audios_Piramides/Chichén_Itzá.mp3",
    "San Pedro Ixcatlan": r"Audios_Piramides/Chichén_Itzá.mp3",
    "San Felipe Jalapa de Díaz": r"Audios_Piramides/Chichén_Itzá.mp3",
    "San Juan Bautista Valle Nacional": r"Audios_Piramides/Chichén_Itzá.mp3",
    "San Lucas Ojitlán": r"Audios_Piramides/Chichén_Itzá.mp3",
}

# --- Agregado: animación de entrada para huipil ---
def animar_huipil_inicio(imagen_original, duracion= 3, pasos=20):
    """
    Muestra una animación donde el huipil aparece en grande y se reduce a su tamaño normal.
    """
    max_size = 700  # Tamaño inicial grande
    min_size = objectRadius * 2  # Tamaño normal
    step = (max_size - min_size) / pasos
    center_x = SCREEN_WIDTH // 2
    center_y = SCREEN_HEIGHT // 2

    for i in range(pasos):
        size = int(max_size - step * i)
        frame = pygame.transform.scale(imagen_original, (size, size))
        x = center_x - size // 2
        y = center_y - size // 2

        screen.blit(background_image, (0, 0))

        # 🔁 DIBUJAR LOS HUIPILES YA COLOCADOS
        for imagen, pos in huipiles_fijos:
            screen.blit(imagen, pos)

        screen.blit(frame, (x, y))
        pygame.display.update()
        pygame.time.delay(int(duracion * 1000 / pasos))



# --- Modificación en reset_game(): efecto visual de entrada aplicado ---
def reset_game():
    global start_time, current_audio, ballFrame

    # Obtener la pirámide actual y cargar su imagen
    current_pyramid = get_current_pyramid()
    imagen_original = pygame.image.load(imagenes_piramide[current_pyramid]).convert_alpha()

    # Reposicionar y detener movimiento antes de la animación
    moving_ball.body.position = (700, 100)
    moving_ball.body.velocity = (0, 0)

    # Animación de entrada
    animar_huipil_inicio(imagen_original)

    # Asignar imagen final escalada
    ballFrame = pygame.transform.scale(imagen_original, (objectRadius * 2, objectRadius * 2))

    # Iniciar el temporizador
    start_time = time.time()

    # Cargar y reproducir el audio
    if current_pyramid in audios_piramides:
     pygame.mixer.music.stop()
    pygame.mixer.music.load(audios_piramides[current_pyramid])
    pygame.mixer.music.play(-1)
    current_audio = current_pyramid



# Rectángulos de colisión para cada huipil
areas2 = {
    "San Felipe Usila": pygame.Rect(495, 600, 65, 150),        # Parte inferior izquierda (verde olivo)
    "San Miguel Soyaltepec": pygame.Rect(495, 335, 110, 110),   # Arriba a la izquierda (azul claro)
    "San Juan Bautista Tuxtepec": pygame.Rect(680, 450, 110, 100),  # Centro-derecha (verde limón)
    "San Pedro Ixcatlan": pygame.Rect(440, 408, 70, 70),      # Parte derecha centro (amarillo)
    "San Felipe Jalapa de Díaz": pygame.Rect(440, 485, 100, 70),  # Centro (aqua)
    "San Juan Bautista Valle Nacional": pygame.Rect(565, 650, 110, 100),  # Abajo al centro (morado oscuro)
    "San Lucas Ojitlán": pygame.Rect(560, 480, 110, 100),      # Zona gris central
}

# Textos renderizados para cada huipil
pyramid_texts = {
    "San Felipe Usila": render_text("San Felipe Usila", 60, (0, 0, 0)),
    "San Miguel Soyaltepec": render_text("San Miguel Soyaltepec", 60, (0, 0, 0)),
    "San Juan Bautista Tuxtepec": render_text("San Juan Bautista Tuxtepec", 60, (0, 0, 0)),
    "San Pedro Ixcatlan": render_text("San Pedro Ixcatlan", 60, (0, 0, 0)),
    "San Felipe Jalapa de Díaz": render_text("San Felipe Jalapa de Díaz", 60, (0, 0, 0)),
    "San Juan Bautista Valle Nacional": render_text("San Juan Bautista Valle Nacional", 60, (0, 0, 0)),
    "San Lucas Ojitlán": render_text("San Lucas Ojitlán", 60, (0, 0, 0)),
}

# Imágenes asociadas a cada huipil
imagenes_piramide = {
    "San Felipe Usila": "imagenes/PAPALOAPAN/usila.png",
    "San Miguel Soyaltepec": "imagenes/PAPALOAPAN/soyaltepec.png",
    "San Juan Bautista Tuxtepec": "imagenes/PAPALOAPAN/tuxtepec.png",
    "San Pedro Ixcatlan": "imagenes/PAPALOAPAN/ixc.png",
    "San Felipe Jalapa de Díaz": "imagenes/PAPALOAPAN/jalapa.png",
    "San Juan Bautista Valle Nacional": "imagenes/PAPALOAPAN/valle.png",
    "San Lucas Ojitlán": "imagenes/PAPALOAPAN/ojitlan.png",
}

def mezclar_piramides(areas2):
    # Obtener una lista de las claves (nombres de pirámides) de las áreas
    claves = list(areas2.keys())
    # Mezclar la lista de claves aleatoriamente
    random.shuffle(claves)
    # Crear un nuevo diccionario con el nuevo orden de pirámides
    nuevo_orden = {clave: areas2[clave] for clave in claves}
    return nuevo_orden

areas = mezclar_piramides(areas2)
pygame.mixer.music.load(audios_piramides[list(areas.keys())[0]])
pygame.mixer.music.play(0)  # Solo se reproduce una vez


# Lista de nombres de pirámides para mantener el orden de juego
pyramid_names = list(areas.keys())
current_pyramid_index = 0  # Índice de la pirámide actual

# Función para mezclar el orden de las pirámides aleatoriamente


# Nuevas variables globales para el estado del juego
# Indica si la pirámide ha sido soltada
pyramid_released = False
release_time = None  # Guarda el tiempo cuando la pirámide fue soltada
waiting_for_result = False  # Indica si se está esperando un resultado tras soltar la pirámide
huipiles_colocados = []
huipiles_fijos = []  # Lista de (imagen, posición) de huipiles ya colocados


piramides_correctas = 0  # Contador de pirámides correctas
total_piramides = len(areas)  # Total de pirámides en el juego
current_audio = None  # Audio actual que está sonando para una pirámide

# Inicializar sonidos usando pygame.mixer para efectos de sonido
pygame.mixer.init()
sonido_correcto = pygame.mixer.Sound("sonido/correcto.mp3")  # Sonido para acierto
sonido_incorrecto = pygame.mixer.Sound("sonido/incorrecto.mp3")  # Sonido para error
sonido_centrar = pygame.mixer.Sound("sonido/centrar.mp3")  # Sonido para error


# Función para renderizar el texto con el tamaño de fuente y color especificado


# Función para obtener la pirámide actual basándose en el índice
def get_current_pyramid():
    return pyramid_names[current_pyramid_index]

# Función para reiniciar el juego después de una colisión correcta
def reset_game():
    global start_time, current_audio, ballFrame

    moving_ball.body.position = (700, 100)  # Restablece la posición de la pirámide
    moving_ball.body.velocity = (0, 0)  # Detiene el movimiento
    start_time = time.time()  # Reinicia el temporizador

    # Obtener la pirámide actual y cargar su imagen
    current_pyramid = get_current_pyramid()
    ballFrame = pygame.image.load(imagenes_piramide[current_pyramid]).convert_alpha()
    ballFrame = pygame.transform.scale(ballFrame, (objectRadius * 2, objectRadius * 2))

    # Cargar y reproducir el audio de la pirámide
    if current_pyramid in audios_piramides:
        pygame.mixer.music.stop()
        pygame.mixer.music.load(audios_piramides[current_pyramid])
        pygame.mixer.music.play(-1)
        current_audio = current_pyramid # Actualiza el audio actual
        pygame.mixer.music.play(-1)



# Función principal para manejar la colisión correcta
def handle_correct_collision():
    global game_over, piramides_correctas, current_pyramid_index, ballFrame, moving_ball

    sonido_correcto.play()
    piramides_correctas += 1
    huipiles_colocados.append(current_pyramid)

    # Guardar la imagen del huipil y su posición final para dejarlo fijo
    imagen = pygame.image.load(imagenes_piramide[current_pyramid]).convert_alpha()
    imagen = pygame.transform.scale(imagen, (objectRadius * 2.8, objectRadius * 2.8))
    posicion_final = (int(moving_ball.body.position[0] - objectRadius), int(moving_ball.body.position[1] - objectRadius))
    huipiles_fijos.append((imagen, posicion_final))  # Guardar la imagen y posición

    if piramides_correctas == total_piramides:
        game_over = True
    else:
        # Avanzar a la siguiente pirámide
        current_pyramid_index += 1
        nueva_piramide = pyramid_names[current_pyramid_index]
        imagen_original = pygame.image.load(imagenes_piramide[nueva_piramide]).convert_alpha()
        animar_huipil_inicio(imagen_original)
        ballFrame = pygame.transform.scale(imagen_original, (objectRadius * 2.7, objectRadius * 2.7))

        # Crear una nueva pirámide móvil
        moving_ball = create_ball(round(objectRadius * 1.4), (900, 100), objectColor)
        reset_game()



# Configurar la ventana de Pygame con el tamaño definido
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.NOFRAME)
pygame.display.set_caption("PIRAMIDES DE LA REPUBLICA MEXICANA")  # Título de la ventana





# Cargar la imagen de fondo (un mapa con puntos) y ajustarla al tamaño de la pantalla
background_image = pygame.image.load('fondo/4.png').convert()  # Carga la imagen de fondo
background_image = pygame.transform.scale(background_image, (SCREEN_WIDTH, SCREEN_HEIGHT))  # Escala la imagen al tamaño de la ventana

# Inicializar el espacio de Pymunk para la simulación física
space = pymunk.Space()  # Crear un espacio para la simulación física
static_body = space.static_body  # Cuerpo estático usado para objetos que no se mueven
# Configurar las opciones de dibujo de Pymunk utilizando Pygame
draw_options = pymunk.pygame_util.DrawOptions(screen)  # Opciones de dibujo para Pygame
# Ajustar los colores de las opciones de dibujo para evitar que se dibujen las colisiones y restricciones
draw_options.collision_point_color = (0, 0, 0, 0)  # Desactiva el color de los puntos de colisión (transparente)
draw_options.constraint_color = (0, 0, 0, 0)  # Desactiva el color de las restricciones (transparente)

# Definir los bordes del área de simulación con líneas estáticas
lines = [
    [(0, 0), (0, SCREEN_HEIGHT)],  # Línea desde la esquina superior izquierda hacia abajo
    [(0, SCREEN_HEIGHT), (SCREEN_WIDTH, SCREEN_HEIGHT)],  # Línea desde la parte inferior izquierda a la derecha
    [(SCREEN_WIDTH, SCREEN_HEIGHT), (SCREEN_WIDTH, 0)],  # Línea desde la parte inferior derecha hacia arriba
    [(SCREEN_WIDTH, 0), (0, 0)]  # Línea desde la esquina superior derecha hacia la izquierda
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

# Crear líneas físicas en el espacio
for c in lines:
    create_line(c[0], c[1], 0.0)
# Inicializar variables de las pirámides y la bola en movimiento
handsShapes = [None, None]
# en esta parte se cambia la pocision en la cual se desea que se inicialice la primera piramide
moving_ball = create_ball(round(objectRadius * 1.4), (900, 100),
                          objectColor)  # Inicializar la primera pirámide en una posición inicial
frametick = 0
frameCount = 0
# Obtener el primer nombre en el diccionario 'areas'
primera_piramide = list(areas.keys())[0]
print(primera_piramide)
# Buscar la dirección en 'imagenes_piramide'
direccion_imagen = imagenes_piramide.get(primera_piramide, "Imagen no encontrada")
# Cargar la imagen de la pirámide con transparencia
objectreadfile = direccion_imagen
pic = pygame.image.load(objectreadfile).convert_alpha()
animar_huipil_inicio(pic)
pic = pygame.transform.scale(pic, (objectRadius * 2, objectRadius * 2))
ballFrame = pic

# clock
clock = pygame.time.Clock()
FPS = 30
# colours
BG = (0, 0, 0)
runGame = True

makefullscreen = True

print("Ejecutando...")

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
# Configuración de la captura de video (cámara)
if makeoptimize:
    cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)  # Optimización para Windows
else:
    cap = cv2.VideoCapture(0)  # Capturar video
# hola
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1300)  # Establecer ancho de video
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 850)  # Establecer altura de video

area_surfaces = {}

# Función para calcular la distancia entre dos puntos
def calc_distance(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Inicializar el temporizador
start_time = time.time()
time_limit = 200  # 100 segundos para cada pirámide
game_over = False

# Iniciar el módulo de detección de manos de MediaPipe con confianza mínima de detección y seguimiento
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
    while cap.isOpened() and runGame and not game_over:
        prueba = True
        clock.tick(FPS)  # Control de velocidad de fotogramas
        space.step(10 / FPS)  # Actualizar el espacio de física de Pymunk

        # Capturar un fotograma de la cámara
        #Para OPENCV
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frameWidth = image.shape[1]
        frameHeight = image.shape[0]
        if invertPic:
            image = cv2.flip(image, 1)  # Invertir imagen si es necesario
        image.flags.writeable = False  # Optimizar imagen
        results = hands.process(image)  # Procesar la imagen para detectar manos
        image.flags.writeable = True  # Volver a hacer la imagen escribible
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convertir imagen de nuevo a BGR
        # se cambia el color del marco de la area de interaccion
        cv2.rectangle(image, gcap1, gcap2, (255, 255, 0), 1)
        totalHands = 0

        #DE VALIABLES PARA EL JUEGO
        # Inicializar variables para el control de la pirámide
        pyramid_held = False
        pyramid_released = False
        #DIBJAR NECESIDADES DEL JUEGO
        # Dibujar la pantalla de fondo y la pirámide
        screen.blit(background_image, (0, 0))
        #space.debug_draw(draw_options)  # Dibujar objetos fisicos
        # Dibujar huipiles que ya fueron colocados correctamente
        for imagen, pos in huipiles_fijos:
         screen.blit(imagen, pos)

        # Dibujar la pirámide en su posición actual
        pyramid_pos = (int(moving_ball.body.position[0] - objectRadius), int(moving_ball.body.position[1] - objectRadius))
        # Calcula la posición donde se debe dibujar la pirámide restando su radio a la posición de la bola (centrado).
        screen.blit(ballFrame, pyramid_pos)  # Dibuja la pirámide en pantalla.

        # Obtener la pirámide actual y su texto asociado
        current_pyramid = get_current_pyramid()  # Función que determina qué pirámide se está moviendo.
        pyramid_text = pyramid_texts[current_pyramid]  # Obtiene el texto correspondiente a la pirámide actual.

        text_x = moving_ball.body.position[0] - pyramid_text.get_width() // 2  # Calcula la posición del texto centrado sobre la pirámide.
        text_y = moving_ball.body.position[1] + objectRadius + 5  # Dibuja el texto justo debajo de la pirámide.

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Obtener las posiciones de los dedos relevantes
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Convertir las coordenadas normalizadas a píxeles
                thumb_pixel = mp_drawing._normalized_to_pixel_coordinates(thumb_tip.x, thumb_tip.y, frameWidth,
                                                                          frameHeight)
                index_pixel = mp_drawing._normalized_to_pixel_coordinates(index_tip.x, index_tip.y, frameWidth,
                                                                          frameHeight)

                if thumb_pixel and index_pixel:
                    # Calcular la distancia entre el pulgar e índice
                    thumb_index_distance = calc_distance(thumb_pixel, index_pixel)

                    # Calcular la distancia entre el índice y el dedo medio
                    # index_middle_distance = calc_distance(index_pixel, middle_pixel)

                    # Si el pulgar y el índice están cerca, mover la pirámide
                    if thumb_index_distance < 50 and current_pyramid not in huipiles_colocados:
                        pyramid_held = True
                        moving_ball.body.position = index_pixel  # Mover la pirámide a la posición del índice
                    elif prueba:
                        if not pyramid_released:
                            pyramid_released = True
                            # Registrar el tiempo de liberación
                            release_time = time.time()
                            # Esperar resultado de colisión
                            waiting_for_result = True

        if waiting_for_result and time.time() - release_time >= 2:  # Verifica si han pasado 3 segundos desde que la pirámide fue soltada.
            waiting_for_result = False
            ball_rect = pygame.Rect(moving_ball.body.position[0], moving_ball.body.position[1], 1, 1)  # Crea un rectángulo pequeño en la posición de la pirámide.
            correct_area = areas[current_pyramid]  # Obtiene el área correcta para la pirámide actual.

            if correct_area.colliderect(ball_rect):  # Verifica si el rectángulo de la pirámide está colisionando con el área correcta.
                print(f"¡Colisión correcta con {current_pyramid}!")
                sonido_correcto.play()  # Reproduce el sonido de colisión correcta.
                handle_correct_collision()  # Maneja la colisión correcta, probablemente sumando puntos.
            else:
                collision_detected = False
                for name, area in areas.items():  # Recorre todas las áreas para verificar si hubo colisión incorrecta.
                    if area.colliderect(ball_rect):
                        print(f"Colisión incorrecta. La pirámide correcta es {current_pyramid}")
                        sonido_incorrecto.play()  # Reproduce el sonido de colisión incorrecta.
                        game_over = True  # Termina el juego por colisión incorrecta.
                        collision_detected = True
                        break
                if not collision_detected:
                    print("No se detectó colisión con ninguna área.")  # No hubo colisión con ninguna área.
                    sonido_centrar.play()
                # Reiniciar el estado si la pirámide se mueve durante la espera
        if waiting_for_result and pyramid_held:  # Si la pirámide es movida mientras se espera resultado.
            waiting_for_result = False  # Cancela la espera por resultado.
            pyramid_released = False  # Reinicia el estado de liberación de la pirámide.

        # Dibujar el área correcta de la pirámide
        for pyramid_name, pyramid_rect in areas.items():  # Recorre todas las áreas de pirámides.
            if pyramid_rect.collidepoint(moving_ball.body.position):  # Verifica si la pirámide actual está en su área correcta.
                pyramid_text = pyramid_texts[pyramid_name]  # Actualiza el texto si la pirámide está en su lugar.

        # Crear superficies transparentes para las áreas una vez

        for name, area in areas.items():
            surface = pygame.Surface((area.width, area.height), pygame.SRCALPHA)
            surface.fill((0, 0, 0, 0))  # Completamente transparente

            pygame.draw.rect(surface, (0, 255, 0, 255), (0, 0, area.width, area.height), 2)  # Borde semi-transparente
            area_surfaces[name] = surface
        for name, area in areas.items():
            color = (255, 255, 255, 0) if name == current_pyramid else (255, 255, 255, 0)
            surface = area_surfaces[name].copy()
            pygame.draw.rect(surface, color, (0, 0, area.width, area.height), 2)  # Actualizar el color del borde
            screen.blit(surface, (area.x, area.y))

        pyramid_text = pyramid_texts[current_pyramid]
        text_x = SCREEN_WIDTH // 2 - pyramid_text.get_width() // 2
        text_y = 50
        screen.blit(pyramid_text, (text_x, text_y))

        instruction_text = render_text(f"Lleva la pirámide al estado que creas correspondiente", 30, (255, 255, 255))
        instruction_x = SCREEN_WIDTH // 2 - instruction_text.get_width() // 2
        instruction_y = 10
        screen.blit(instruction_text, (instruction_x, instruction_y))
        # CAMBIO 1
        counter_text = render_text(f"PIRÁMIDES CORRECTAS: {piramides_correctas}/{4}", 30, (255, 255, 255))
        counter_x = SCREEN_WIDTH - counter_text.get_width() - 10
        counter_y = 10
        screen.blit(counter_text, (counter_x, counter_y))

        elapsed_time = time.time() - start_time
        remaining_time = max(0, time_limit - elapsed_time)
        time_text = render_text(f"Tiempo: {int(remaining_time)}s", 30, (0, 0, 255))
        time_x = 10
        time_y = 10
        screen.blit(time_text, (time_x, time_y))
        # CAMBIO 2
        if remaining_time <= 0 or piramides_correctas == 4:
            pygame.mixer.music.pause()
            game_over = True

        if frameCount > 15:
            frameCount = 0  # Reinicia el conteo de fotogramas cada 15 ciclos para optimizar rendimiento.

        if frametick > 0:
            frametick = 0
            frameCount += 1
        frametick += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                runGame = False  # Salir si se cierra la ventana

        # Mostrar la ventana de video si no está en pantalla completa
        if not makefullscreen:
            cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Salir si se presiona 'q'

        pygame.display.update()

    if game_over:
        # screen.fill(BG)
        #screen.blit(background_image, (0, 0))
        # CAMBIO 3
        if piramides_correctas == 4:
            
            game_over_text = render_text("¡FELICIDADES!", 60, (0, 255, 0))
            text_x = SCREEN_WIDTH // 2 - game_over_text.get_width() // 2
            text_y = SCREEN_HEIGHT // 2 - game_over_text.get_height() // 2
            screen.blit(game_over_text, (text_x, text_y))

            congrats_text = render_text("ACERTASTE TODOS LOS HUIPILES", 40, (0, 0, 255))
            congrats_x = SCREEN_WIDTH // 2 - congrats_text.get_width() // 2
            congrats_y = text_y + game_over_text.get_height() + 20
            screen.blit(congrats_text, (congrats_x, congrats_y))

            gift_text = render_text("TOMA TU OBSEQUIO", 40, (255, 255, 0))
            gift_x = SCREEN_WIDTH // 2 - gift_text.get_width() // 2
            gift_y = congrats_y + congrats_text.get_height() + 40
            screen.blit(gift_text, (gift_x, gift_y))

            # print("1")
            try:
                subprocess.Popen(['python', 'menu.py'])
            except Exception as e:
                    print(f"Error al ejecutar menu.py: {e}")
                    sys.exit()
            cap.release()
            cv2.destroyAllWindows()
        else:
            # Definir el texto 'Game Over'
            # Renderizar el texto "Game Over"
            game_over_text = render_text("Game Over", 50, (255, 0, 0))  # Texto "Game Over" con tamaño 50 y color rojo

            # Centrar el texto "Game Over" en pantalla
            text_x = SCREEN_WIDTH // 2 - game_over_text.get_width() // 2
            text_y = SCREEN_HEIGHT // 2 - game_over_text.get_height() // 2

            # Mostrar el texto "Game Over" en pantalla
            screen.blit(game_over_text, (text_x, text_y))

            # Ahora que game_over_text está definido, puedes calcular score_y
            score_y = text_y + game_over_text.get_height() + 5
            # CAMBIO 4
            # Mostrar el texto del puntaje final
            final_score_text = render_text(f"Pirámides: {piramides_correctas}/{4}", 30, (255, 255, 255))
            score_x = SCREEN_WIDTH // 2 - final_score_text.get_width() // 2
            screen.blit(final_score_text, (score_x, score_y))

            # Mostrar instrucciones para salir
            instruction_text = render_text("REGRESANDO AL MENU PRINCIPAL", 30, (218, 0, 167))
            instruction_x = SCREEN_WIDTH // 2 - instruction_text.get_width() // 2
            instruction_y = score_y + final_score_text.get_height() + 10
            screen.blit(instruction_text, (instruction_x, instruction_y))
            try:
                subprocess.Popen(['python', 'menu.py'])
            except Exception as e:
                    print(f"Error al ejecutar menu.py: {e}")
                    sys.exit()
            cap.release()
            cv2.destroyAllWindows()


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