import numpy as np
from skimage import io, color
import streamlit as st

def template_matching_sdc(image_gray, template_gray):
    img_h, img_w = image_gray.shape
    temp_h, temp_w = template_gray.shape

    if temp_h > img_h or temp_w > img_w:
        raise ValueError("El template es más grande que la imagen principal.")

    result = np.zeros((img_h - temp_h + 1, img_w - temp_w + 1))

    for y in range(result.shape[0]):
        for x in range(result.shape[1]):
            # Extrae la región de la imagen del mismo tamaño que el template
            region = image_gray[y:y + temp_h, x:x + temp_w]
            # Calcula la SDC
            sdc = np.sum((region - template_gray) ** 2)
            result[y, x] = sdc
    return result

def load_image(uploaded_image):
    image = io.imread(uploaded_image)
    if image.shape[2] == 4:
        image = image[..., :3]
    return image

def draw_border(image, top_left, width, height, color=(255, 0, 0), thickness=2):
    y, x = top_left
    image[y:y+thickness, x:x+width] = color
    image[y+height-thickness:y+height, x:x+width] = color
    image[y:y+height, x:x+thickness] = color
    image[y:y+height, x+width-thickness:x+width] = color
    return image

st.title("Easter Egg Finder")

# Carga de imágenes
uploaded_image = st.file_uploader("Sube la imagen principal", type=["jpg", "jpeg", "png"])
uploaded_template = st.file_uploader("Sube el template (Easter Egg)", type=["jpg", "jpeg", "png"])

if uploaded_image and uploaded_template:
    try:
        # Cargar y procesar las imágenes
        image_color = load_image(uploaded_image)
        template_color = load_image(uploaded_template)

        # Convertir a escala de grises para procesamiento SDC
        image_gray = color.rgb2gray(image_color)
        template_gray = color.rgb2gray(template_color)

        # Realizar Template Matching
        result = template_matching_sdc(image_gray, template_gray)
        min_loc = np.unravel_index(np.argmin(result), result.shape)
        top_left_y, top_left_x = min_loc
        temp_h, temp_w = template_gray.shape

        # Calcular posiciones de las esquinas
        top_left = (top_left_y, top_left_x)
        top_right = (top_left_y, top_left_x + temp_w)
        bottom_left = (top_left_y + temp_h, top_left_x)
        bottom_right = (top_left_y + temp_h, top_left_x + temp_w)

        # Dibuja el marco en la imagen original a color
        image_with_box = image_color.copy()
        image_with_box = draw_border(image_with_box, (top_left_y, top_left_x), temp_w, temp_h, color=(255, 0, 0), thickness=5)

        # Muestra los resultados
        st.image(image_with_box, caption="Imagen Principal con Easter Egg detectado", use_column_width=True)
        st.image(template_color, caption="Template (Easter Egg)", use_column_width=True)
        st.text(f"Posición encontrada: \n"
                f"Esquina superior izquierda (Y: {top_left_y}, X: {top_left_x})\n"
                f"Esquina superior derecha (Y: {top_right[0]}, X: {top_right[1]})\n"
                f"Esquina inferior izquierda (Y: {bottom_left[0]}, X: {bottom_left[1]})\n"
                f"Esquina inferior derecha (Y: {bottom_right[0]}, X: {bottom_right[1]})")

    except ValueError as ve:
        st.error(f"Error: {ve}")
    except Exception as e:
        st.error(f"Ocurrió un error inesperado: {e}")
