import nbformat

# Nombres de tus archivos
archivos = ['final3.ipynb', 'conclusions_final_model.ipynb']
salida = 'final4.ipynb'

# Leemos el primer notebook
nb_final = nbformat.read(archivos[0], 4)

# Leemos el segundo y añadimos sus celdas al primero
nb_segundo = nbformat.read(archivos[1], 4)
nb_final.cells.extend(nb_segundo.cells)

# Guardamos el resultado
with open(salida, 'w', encoding='utf-8') as f:
    nbformat.write(nb_final, f)

print(f"¡Listo! Archivos fusionados en {salida}")