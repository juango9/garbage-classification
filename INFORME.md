# Informe Técnico: Clasificación Automática de Residuos mediante Deep Learning

## 1. Introducción y Motivación

La gestión de residuos es uno de los desafíos ambientales más relevantes del siglo XXI. La clasificación automática de basura es fundamental para:

- **Automatización de plantas de reciclaje**: Separación rápida y precisa de materiales
- **Optimización de procesos**: Reducción de costos operacionales
- **Sostenibilidad ambiental**: Incremento en tasas de recuperación y reciclaje
- **Aplicaciones IoT**: Integración en sistemas de gestión inteligente de residuos

Este proyecto aborda el problema mediante técnicas de **visión artificial y aprendizaje profundo**, comparando múltiples enfoques desde simples clasificadores lineales hasta redes neuronales convolucionales con transfer learning.

---

## 2. Definición del Problema

### 2.1 Tarea
**Clasificación supervisada de imágenes multiclase** de residuos

**Entrada (X)**: Imágenes RGB de elementos de basura  
**Salida (y)**: Una de 6 categorías de residuos

### 2.2 Clases
1. Cartón
2. Vidrio
3. Metal
4. Papel
5. Plástico
6. Basura Mixta (Trash)

---

## 3. Dataset y Metodología

### 3.1 Datos
- **Fuente**: Garbage Classification dataset (Kaggle)
- **Volumen**: ~2,500 imágenes
- **Formato**: JPEG, imágenes RGB variadas
- **Clases**: 6 (balanceadas aproximadamente)

### 3.2 División de Datos
Se aplicó **estratificación** para conservar proporciones de clases:

| Conjunto     | Porcentaje | Propósito |
|-------------|-----------|----------|
| Entrenamiento | 80%       | Ajuste de parámetros |
| Validación    | 10%       | Tuning e hiperparámetros |
| Test          | 10%       | Evaluación final (no visto durante entrenamiento) |

### 3.3 Métricas de Evaluación

Se utilizaron múltiples métricas complementarias:

$$\text{Exactitud (Accuracy)} = \frac{\text{Predicciones correctas}}{\text{Total de predicciones}}$$

$$\text{Precisión} = \frac{TP}{TP + FP}$$

$$\text{Recall (Sensibilidad)} = \frac{TP}{TP + FN}$$

$$F1 = 2 \cdot \frac{\text{Precisión} \cdot \text{Recall}}{\text{Precisión} + \text{Recall}}$$

Donde TP = Verdaderos Positivos, FP = Falsos Positivos, FN = Falsos Negativos

### 3.4 Evaluación de Reproducibilidad (multi-semilla)

Además de la evaluación principal por split fijo, se incorporó una validación de robustez para los mejores backbones:

- Notebook `notebooks/Evaluacion_modelo8.ipynb` (EfficientNetB0)
- Notebook `notebooks/Evaluacion_modelo9.ipynb` (DenseNet121)

En ambos casos se entrenó y evaluó el modelo con cuatro semillas (`42`, `123`, `7`, `99`) y se reportó:

- Accuracy de entrenamiento
- Accuracy de validación
- Accuracy de test
- Media entre semillas

Este análisis permite separar dos aspectos distintos: rendimiento máximo puntual y estabilidad del modelo ante variaciones de inicialización/split aleatorio.

---

## 4. Enfoques Experimentados y Evolución

Se siguió una **estrategia progresiva** desde modelos simples a complejos:

### 4.1 Modelos Clásicos de Machine Learning

#### Modelo 1: Regresión Logística
- **Parámetros**: 294,918
- **Características**: Flattening de imágenes + Regresión Logística
- **Resultados**:
  - Train Accuracy: **100.00%**
  - Val Accuracy: **40.00%** ❌ (Severo overfitting)
  - Test Accuracy: **45.00%**
- **Conclusión**: Sobreentrena catastroficamente sin poder generalizar

#### Modelo 2: Random Forest (GridSearchCV)
- **Parámetros**: 65,056
- **Características**: Características manuales + Best Hyperparameters
- **Resultados**:
  - Train Accuracy: **99.80%**
  - Val Accuracy: **63.24%** ⚠️ (Mejor que logística, aún con gap)
  - Test Accuracy: **66.40%**
- **Conclusión**: Mejor generalización pero limitado por características hand-crafted

### 4.2 Redes Neuronales Convolucionales (CNNs) Desde Cero

#### CNN Más Básica
- **Parámetros**: 210
- **Arquitectura**: Minimalista (pocas capas)
- **Resultados**:
  - Train Accuracy: **37.00%**
  - Val Accuracy: **38.00%**
  - Test Accuracy: **38.00%**
- **Conclusión**: Demasiado simple, insuficiente capacidad

#### Modelo 1 (CNN pequeña)
- **Parámetros**: 1,094
- **Mejora**: Arquitectura mejorada respecto a la básica
- **Resultados**:
  - Train Accuracy: **61.65%**
  - Val Accuracy: **60.87%**
  - Test Accuracy: **64.43%** ⬆️
- **Conclusión**: Aprendizaje inicial consistente

#### Modelo 2 (CNN media)
- **Parámetros**: 94,022
- **Mejora**: Capas adicionales, mejor extracción de características
- **Resultados**:
  - Train Accuracy: **91.93%**
  - Val Accuracy: **87.35%** ✅
  - Test Accuracy: **83.00%**
- **Conclusión**: **Primer modelo con buena generalización** (~83% test)

#### Modelo 3 (CNN compleja)
- **Parámetros**: 389,958
- **Mejora**: Arquitectura más profunda y refinada
- **Resultados**:
  - Train Accuracy: **99.80%**
  - Val Accuracy: **88.54%** ✅
  - Test Accuracy: **85.38%** ⬆️
- **Conclusión**: Más profundidad mejora a ~85%, pero comienza overfitting

#### Modelo 4 (CNN experimental)
- **Parámetros**: 409,286
- **Observación**: Arquitectura alternativa
- **Resultados**:
  - Train Accuracy: **79.96%**
  - Val Accuracy: **70.75%** ⚠️
  - Test Accuracy: **77.08%**
- **Conclusión**: Rendimiento inferior; arquitectura menos óptima

#### Modelo 5 (CNN refinada)
- **Parámetros**: 474,822
- **Mejora**: Ajustes a partir de Modelo 3
- **Resultados**:
  - Train Accuracy: **89.76%**
  - Val Accuracy: **84.58%**
  - Test Accuracy: **81.03%** ⬇️
- **Conclusión**: Regresión respecto a Modelo 3

#### Modelo 6 (ResNet50 - Transfer Learning)
- **Parámetros**: 24,147,078
- **Cambio metodológico**: **Primera red preentrenada** (ImageNet)
- **Arquitectura**: ResNet50 + fine-tuning
- **Resultados**:
  - Train Accuracy: **99.90%**
  - Val Accuracy: **89.33%** ✅
  - Test Accuracy: **87.35%** ⬆️⬆️
- **Conclusión**: **Mejor rendimiento hasta ahora**, salto de +4pp vs CNN

#### Modelo 7 (ResNet50 - base congelada)
- **Parámetros**: 24,147,078
- **Cambio metodológico**: Transfer learning con backbone congelado
- **Arquitectura**: ResNet50 + cabeza densa (sin fine-tuning del backbone)
- **Resultados**:
  - Train Accuracy: **99.85%**
  - Val Accuracy: **88.54%**
  - Test Accuracy: **86.56%**
- **Conclusión**: Similar a ResNet50 fine-tuned, pero ligeramente inferior en test

#### Modelo 8 (EfficientNetB0 - Transfer Learning)
- **Parámetros**: 4,427,177
- **Cambio metodológico**: Backbone más eficiente con fine-tuning
- **Arquitectura**: EfficientNetB0 + regularización y augmentation opcional
- **Resultados**:
  - Train Accuracy: **99.90%**
  - Val Accuracy: **92.09%** ✅
  - Test Accuracy: **89.72%** ⬆️⬆️
- **Conclusión**: Mejora clara respecto a ResNet50 con muchos menos parámetros

#### Modelo 9 (DenseNet121 - Transfer Learning)
- **Parámetros**: 7,349,574
- **Cambio metodológico**: Backbone denso con fine-tuning
- **Arquitectura**: DenseNet121 + cabeza densa
- **Resultados**:
  - Train Accuracy: **99.60%**
  - Val Accuracy: **92.89%** ✅
  - Test Accuracy: **90.51%** 🏆
- **Conclusión**: **Mejor modelo global del proyecto**, máximo rendimiento en test

### 4.3 Análisis Comparativo de Evolución

```
Evolución del Test Accuracy:
═══════════════════════════════════════════
45.0%  Logistic Regression          │
66.4%  Random Forest                ││
38.0%  CNN Básica                   │
64.4%  CNN Modelo 1                 ││
83.0%  CNN Modelo 2         ════════╪════
85.4%  CNN Modelo 3         ════════╪═════
77.1%  CNN Modelo 4         ═══════╪════
81.0%  CNN Modelo 5         ════════╪═══
87.4%  CNN Modelo 6 (ResNet50)      ════════╪══════
86.6%  CNN Modelo 7 (ResNet50 freeze)═══════╪═════
89.7%  CNN Modelo 8 (EfficientNetB0)═══════╪═══════
90.5%  CNN Modelo 9 (DenseNet121)   ════════╪════════ ⭐ MEJOR
```

**Hito clave**: El salto a transfer learning y la comparación de backbones (+5.1pp sobre CNN3)

---

## 5. Resultados Finales Resumidos

### 5.1 Tabla Comparativa Completa

| Modelo | Parametros | Enfoque | Train Acc | Val Acc | Test Acc | Overfitting |
|---------|-----------|---------|-----------|---------|----------|------------|
| Logistic Regression | 294K | Clásico | 100.00% | 40.00% | 45.00% | Severo ❌ |
| Random Forest | 65K | Clásico | 99.80% | 63.24% | 66.40% | Alto ⚠️ |
| CNN Básica | 210 | DL simple | 37.00% | 38.00% | 38.00% | Subajuste ❌ |
| CNN Modelo 1 | 1K | DL pequeña | 61.65% | 60.87% | 64.43% | Bajo |
| CNN Modelo 2 | 94K | DL media | 91.93% | 87.35% | 83.00% | Bajo ✅ |
| CNN Modelo 3 | 390K | DL profunda | 99.80% | 88.54% | 85.38% | Moderado ⚠️ |
| CNN Modelo 4 | 409K | DL alternativa | 79.96% | 70.75% | 77.08% | Alto ⚠️ |
| CNN Modelo 5 | 475K | DL refinada | 89.76% | 84.58% | 81.03% | Bajo |
| CNN Modelo 6 (ResNet50) | 24.1M | Transfer Learning | 99.90% | 89.33% | 87.35% | Moderado |
| CNN Modelo 7 (ResNet50 freeze) | 24.1M | Transfer Learning | 99.85% | 88.54% | 86.56% | Moderado |
| CNN Modelo 8 (EfficientNetB0) | 4.4M | Transfer Learning | 99.90% | 92.09% | 89.72% | Bajo ✅ |
| **CNN Modelo 9 (DenseNet121)** | **7.3M** | **Transfer Learning** | **99.60%** | **92.89%** | **90.51%** | **Bajo** ⭐ |

### 5.2 Ganancia de Rendimiento

**Mejora progresiva**:
- De Regresión Logística a Random Forest: +21.4pp *(métodos clásicos)*
- De Random Forest a CNN Modelo 2: +16.6pp *(primeras redes)*
- De CNN Modelo 2 a CNN Modelo 3: +2.4pp *(profundidad)*
- De CNN Modelo 3 a CNN Modelo 8: +4.3pp *(transfer learning eficiente)*
- De CNN Modelo 8 a CNN Modelo 9: +0.8pp *(mejor backbone en este dataset)*

**Total desde el inicio**: **45.5 puntos porcentuales** (45.0% → 90.5%)

### 5.3 Evaluación por Semillas (Modelos 8 y 9)

#### Modelo 8 (EfficientNetB0)

| Seed   |   Train Accuracy |   Val Accuracy |   Test Accuracy |
|:-------|-----------------:|---------------:|----------------:|
| 42     |           0.9990 |         0.9249 |          0.9012 |
| 123    |           0.9980 |         0.9012 |          0.8617 |
| 7      |           0.9985 |         0.9249 |          0.8775 |
| 99     |           0.9985 |         0.8893 |          0.8656 |
| Media  |           0.9985 |         0.9101 |          0.8765 |

#### Modelo 9 (DenseNet121)

| Seed   |   Train Accuracy |   Val Accuracy |   Test Accuracy |
|:-------|-----------------:|---------------:|----------------:|
| 42     |            0.999 |         0.8972 |          0.9012 |
| 123    |            0.999 |         0.8893 |          0.8854 |
| 7      |            0.999 |         0.8893 |          0.8696 |
| 99     |            0.999 |         0.8933 |          0.8972 |
| Media  |            0.999 |         0.8923 |          0.8883 |

### 5.4 Lectura Conjunta: Rendimiento vs Robustez

- En evaluación puntual por split principal, **Modelo 9** obtiene el mejor test (**90.51%**).
- En evaluación multi-semilla, **Modelo 9** también logra mejor media en test (**88.83% vs 87.65%**).
- **Modelo 8** muestra mejor media en validación (**91.01% vs 89.23%**), por lo que es competitivo en estabilidad de validación.
- Ambos modelos son de alta calidad; la elección final depende de si se prioriza máxima accuracy en test o regularidad en validación.

---

## 6. Análisis e Interpretación

### 6.1 Lecciones de Machine Learning Clásico

1. **Regresión Logística falla** con imágenes crudas (píxeles como features)
2. **Random Forest mejora** (66%) pero insuficiente para fotos complejas
3. **Ambos padecen** de falta de feature engineering adecuada

### 6.2 Impacto de Arquitectura DL

1. **Complejidad es necesaria** pero no suficiente:
   - CNN Básica (210 params) → Subajuste (38%)
   - CNN Modelo 3 (390K) → Mejor pero con límite (85.4%)

2. **Profundidad ayuda** (Modelo 1 → Modelo 3: 64% → 85%)
   - Pero trae consigo riesgo de overfitting

3. **Tuning de arquitectura** es crítico:
   - Modelo 4 (con 409K params) es **peor** que Modelo 3 (390K)
   - Sugiere que los detalles arquitectónicos importan más que solo los parámetros

### 6.3 Transfer Learning Como Punto Quiebre

Los modelos preentrenados en ImageNet (ResNet50, EfficientNetB0 y DenseNet121) superan de forma consistente a las CNN entrenadas desde cero:
- **ResNet50** consolida el salto inicial a transfer learning (87.35% test)
- **EfficientNetB0** mejora el compromiso rendimiento/parámetros (89.72% con 4.4M)
- **DenseNet121** logra el mejor resultado global (**90.51% test**)

Esto confirma el beneficio de reutilizar **características universales** y ajustar finamente para el dominio de residuos.

### 6.4 Generalización y Robustez

Comparando los dos mejores modelos:

- **CNN Modelo 9 (DenseNet121)**:
  - Train: 99.60%
  - Val: 92.89%
  - Test: 90.51%
  - Gap train-val: 6.71pp

- **CNN Modelo 8 (EfficientNetB0)**:
  - Train: 99.90%
  - Val: 92.09%
  - Test: 89.72%
  - Gap train-val: 7.81pp

En ambos casos la brecha es menor que en varias configuraciones previas y, adicionalmente, la evaluación por semillas confirma comportamiento estable, especialmente en test para Modelo 9.

---

## 7. Conclusiones

### 7.1 Conclusión Principal

Se alcanzó un **clasificador de 90.51% de exactitud** usando DenseNet121 con transfer learning (evaluación principal), comparado con un baseline de 45% (regresión logística). Además, en evaluación multi-semilla, DenseNet121 mantiene la mejor media en test (**88.83%**), reforzando que no se trata de un resultado aislado.

### 7.2 Hallazgos Clave

1. **Los métodos clásicos (ML) son insuficientes** para imágenes de residuos complejos (máx 66%)

2. **Las CNNs desde cero logran ~85%**, pero requieren:
   - Arquitectura adecuada
   - Suficientes datos
   - Regularización contra overfitting

3. **Transfer Learning es superior** para este problema (hasta **90.51%**):
   - Aprovecha características de ImageNet
   - Mejor generalización
   - Reducción del gap train-val

4. **Relación Parámetros ≠ Rendimiento**:
  - Más parámetros no garantiza mejor resultado (EfficientNetB0 y DenseNet121 superan a ResNet50 con menos parámetros)
  - La arquitectura y el tuning son críticos

5. **La evaluación con múltiples semillas es clave**:
  - Permite medir robustez y no solo el mejor caso
  - DenseNet121 destaca por rendimiento medio en test

### 7.3 Ventajas de la Solución
- ✅ **Automatizable**: Puede procesarse en tiempo real
- ✅ **Escalable**: Aplicable a otras categorías de residuos
- ✅ **Reproducible**: Código modular y documentado
- ✅ **Información útil**: Generación de métricas por clase

### 7.4 Limitaciones y Futuro

**Limitaciones actuales**:
- Dataset pequeño (~2500 imágenes)
- Imágenes de calidad variable
- 90.51% es un resultado sólido, pero aún puede ser insuficiente para escenarios críticos (objetivo ~95%+)

**Mejoras futuras**:
- ⬆️ Aumentar dataset (data augmentation, web scraping)
- ⬆️ Usar modelos más grandes (EfficientNet, DenseNet)
- ⬆️ Ensembles de modelos
- ⬆️ Self-supervised learning (como se reporta en literatura: 97%)
- ⬆️ Edge deployment optimization (MobileNetV2 para dispositivos)
- ⬆️ Reportar desviación estándar en evaluación por semillas para complementar la media

### 7.5 Impacto Potencial
Con un accuracy de 90.51%, el sistema tiene un **error de clasificación del 9.49%**, lo que implica que en una planta de 1000 residuos diarios:
- Correctamente clasificados: **905.1**
- Requieren revisión manual: **94.9**

Este es un nivel aceptable para automatización asistida, donde humanos supervisor detectan errores.

---

## 8. Recomendaciones

1. **Para Producción**:
  - Usar DenseNet121 / EfficientNetB0 o arquitectura similar con transfer learning
   - Implementar data augmentation
   - Crear pipeline de reentrenamiento mensual
   - Integrar mecanismo de rechazo (confianza baja → revisión manual)

2. **Para Investigación**:
   - Experimentar con Vision Transformers (ViT)
   - Explorar self-supervised learning
   - Analizar confusiones por pares de clases
   - Estudiar impacto de iluminación y ángulo

3. **Para Datos**:
   - Recolectar más imágenes (~10K)
   - Balancear clases si hay sesgos
   - Anotar metadatos (iluminación, ángulo, tamaño)

---

## 9. Referencias

- Dataset: Kaggle Garbage Classification (https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)
- Paper Relacionado: "Waste Classification with Deep Convolutional Neural Networks" (2020)
- Frameworks: TensorFlow/Keras, scikit-learn, OpenCV
- Hardware: GPU (para entrenamientos eficientes)

---

**Fecha de Finalización**: Proyecto completado  
**Autor**: Proyecto de Deep Learning  
**Versión del Documento**: 1.0
