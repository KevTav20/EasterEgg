import numpy as np
from skimage import io, color, transform
import streamlit as st
from io import BytesIO
from PIL import Image

def template_matching_sdc(image_gray, template_gray, threshold):
    img_h, img_w = image_gray.shape
    temp_h, temp_w = template_gray.shape
    result = np.zeros((img_h - temp_h + 1, img_w - temp_w + 1))

    for y in range(result.shape[0]):
        for x in range(result.shape[1]):
            region = image_gray[y:y + temp_h, x:x + temp_w]
            sdc = np.sum((region - template_gray) ** 2) / (temp_h * temp_w)
            result[y, x] = sdc

    min_val = np.min(result)
    if min_val <= threshold:
        return np.unravel_index(np.argmin(result), result.shape)
    else:
        return None

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

def scale_and_rotate_template(template, scales, rotations):
    templates = []
    for scale in scales:
        scaled_template = transform.rescale(template, scale, anti_aliasing=True, channel_axis=None)
        for angle in rotations:
            rotated_template = transform.rotate(scaled_template, angle)
            templates.append(rotated_template)
    return templates

st.title("Buscador de Easter Eggs")

uploaded_image = st.file_uploader("Sube la imagen principal", type=["jpg", "jpeg", "png"])
uploaded_template = st.file_uploader("Sube el template (Easter Egg)", type=["jpg", "jpeg", "png"])

if uploaded_image and uploaded_template:
    try:
        # Cargar y procesar las imágenes
        image_color = load_image(uploaded_image)
        template_color = load_image(uploaded_template)
        image_gray = color.rgb2gray(image_color)
        template_gray = color.rgb2gray(template_color)

        # Disposición en columnas
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_color, caption="Imagen Principal", use_column_width=True)
        with col2:
            st.image(template_color, caption="Template (Easter Egg)", use_column_width=True)

        # Parámetros del usuario en una sección expandible
        with st.expander("Configuración Avanzada"):
            scales = st.multiselect("Escalas para probar", [0.5, 1, 1.5, 2], default=[1])
            rotations = st.multiselect("Rotaciones (grados) para probar", [0, 45, 90, 135, 180], default=[0])
            threshold = st.slider("Umbral de detección", min_value=0.0, max_value=1.0, value=0.05)
            border_color = st.color_picker("Color del borde", "#FF0000")
            border_thickness = st.slider("Grosor del borde", min_value=1, max_value=10, value=5)

        # Crear variantes del template
        templates = scale_and_rotate_template(template_gray, scales, rotations)

        # Indicador de estado
        status_placeholder = st.empty()
        status_placeholder.text("Procesando...")

        # Realizar Template Matching en cada variante
        best_match = None
        for temp_variant in templates:
            match_location = template_matching_sdc(image_gray, temp_variant, threshold)
            if match_location:
                top_left_y, top_left_x = match_location
                temp_h, temp_w = temp_variant.shape
                best_match = (top_left_y, top_left_x, temp_w, temp_h)
                break

        if best_match:
            status_placeholder.success("¡Easter Egg encontrado!")
            top_left_y, top_left_x, temp_w, temp_h = best_match

            # Dibujar el borde alrededor del Easter Egg detectado
            image_with_box = draw_border(
                image_color.copy(),
                (top_left_y, top_left_x),
                temp_w,
                temp_h,
                color=tuple(int(border_color[i:i+2], 16) for i in (1, 3, 5)),
                thickness=border_thickness
            )

            st.image(image_with_box, caption="Easter Egg Detectado", use_column_width=True)

            # Convertir image_with_box a formato PIL para descargar
            pil_image = Image.fromarray((image_with_box * 255).astype(np.uint8))  # Convertir a 8-bit para PIL
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()

            # Mostrar las coordenadas de la detección
            st.text(f"Coordenadas detectadas:\n"
                    f"Esquina Superior Izquierda (Y: {top_left_y}, X: {top_left_x})\n"
                    f"Esquina Superior Derecha (Y: {top_left_y}, X: {top_left_x + temp_w})\n"
                    f"Esquina Inferior Izquierda (Y: {top_left_y + temp_h}, X: {top_left_x})\n"
                    f"Esquina Inferior Derecha (Y: {top_left_y + temp_h}, X: {top_left_x + temp_w})")

            # Opción para descargar la imagen con la detección
            st.download_button(
                label="Descargar imagen con detección",
                data=img_bytes,
                file_name="easter_egg_detectado.png",
                mime="image/png"
            )
        else:
            status_placeholder.warning("No se encontró ninguna coincidencia con los parámetros seleccionados.")
    except ValueError as ve:
        st.error(f"Error: {ve}")
    except Exception as e:
        st.error(f"Ocurrió un error inesperado: {e}")
