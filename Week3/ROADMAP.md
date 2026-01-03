# Work to do

## 0 - Recolectar resultados best config entregas anteriores con nuevo dataset -> DONE! (ARNAU)

1. Preparar codigo para que vaya con al wandb
2. Mejor modelo Week 2

## 1 - Fine Tune con nuestra arquitectura (DenseNet-121) -> DONE! (ADRIÀ)

1. Cambiar ultima capa para que saque 8 outputs
2. Congelar todo menos ultima capa
3. Entrenar hasta que la loss pare de mejorar
4. Descongelar la capa anterior
5. Repetir hasta descongelar todas las capas de los ultimos dos bloques

## 2 - Set a new model

1. Probar a quitar capas (quitar capas una a una, quitar ultimo bloque)
2. Otras cosas?

## 3 - Probar a entrenar con pocos datos

Esto se tiene que hacer en todos

## 4 - Data augmentation

1. Probar con data augmentation VS sin data augmentation
2. Probar diferentes tipos de augmentations (quitar y poner algunos que sean relevantes): flips, zooms, rescales, noise + los de la semana pasada

## 5 - Hyperparameter optimization - Topology

1. Fintetuning con dropout
2. Batch norm (ya tiene)
3. Regularizacion (l2)

## 6 - Hyperparameter optimization - Hyperparameters

1. Random hyperparameter search con Sweeps del wandb

## 7 - Report

1. Visualizaciones con el GradCam
