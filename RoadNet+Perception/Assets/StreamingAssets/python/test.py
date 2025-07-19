import base64

def guardar_imagen_desde_string(bin_str: str, nombre_archivo: str) -> None:
    """
    Convierte una cadena que representa datos binarios de imagen (ejemplo: base64)
    en un archivo de imagen válido en disco.
    
    Parámetros:
      - bin_str: str — cadena con datos binarios codificados en base64.
      - nombre_archivo: str — ruta + nombre de salida, ej. 'frame.png'.
    """
    # Si la cadena viene con prefijo como 'data:image/png;base64,xxx', lo removemos:
    if ',' in bin_str:
        _, bin_str = bin_str.split(',', 1)
    
    # Decodificamos de base64 a bytes
    img_bytes = base64.b64decode(bin_str)
    
    # Escribimos en modo binario
    with open(nombre_archivo, 'wb') as f:
        f.write(img_bytes)
