# Presentación: Clasificación Automática de Residuos 🗑️
## Duración: 5-6 minutos

---

## SLIDE 1: EL PROBLEMA
### ¿Por qué es importante?

Cada año se generan **2.12 mil millones de toneladas de residuos** a nivel mundial.
- ❌ **Manual**: Lento, caro y poco preciso
- ✅ **Automático**: Rápido, escalable y consistente

🎯 **Objetivo**: Crear un sistema inteligente que clasifique basura en **6 categorías**:
- Cartón | Vidrio | Metal | Papel | Plástico | Basura Mixta

---

## SLIDE 2: NUESTRA SOLUCIÓN
### Visión por Computadora + Deep Learning

```
Imagen de Residuo
        ↓
  Procesamiento
        ↓
Red Neural Profunda
        ↓
Categoría Predicha
(+ Confianza)
```

**Dataset**: ~2,500 imágenes reales de 6 clases  
**División**: 80% entrenamiento, 10% validación, 10% test  
**Métrica**: Accuracy (¿qué % clasifica correctamente?)

---

## SLIDE 3: EVOLUCIÓN DE MODELOS
### Desde lo simple a lo sofisticado

| Enfoque | Mejor Accuracy | Conclusión |
|---------|----------------|-----------|
| **Regresión Logística** | 45% ❌ | Demasiado simple |
| **Random Forest** | 66% ⚠️ | Mejor pero limitado |
| **CNN desde cero** | 85% ✅ | Funciona bien |
| **EfficientNetB0** (Transfer Learning) | **89.7%** ✅ | Gran salto |
| **DenseNet121** (Transfer Learning) | **90.5%** ⭐ | **Lo mejor** |

📊 **Mejora total**: De 45% → 90.5%
= **Reducción de errores del 82.7%**

---

## SLIDE 4: ¿POR QUÉ GANA DENSENET121?

### Transfer Learning = Reutilizar conocimiento

Los modelos backbone fueron preentrenados con millones de imágenes (ImageNet)  
→ Ya saben detectar bordes, texturas y formas genéricas

Luego se "ajusta fino" con nuestras imágenes de basura  
→ Combina lo mejor: capacidad general + especialización

**Resultado**: DenseNet121 logra la máxima accuracy (**90.5%**) y excelente generalización

---

## SLIDE 5: MÉTRICAS FINALES (EVALUACIÓN PRINCIPAL)

### DenseNet121 en Test Set (nunca visto)

```
Accuracy:          90.51% ✅
        ├─ Predicciones correctas: 905 de 1000
        └─ Errores: 95 de 1000

Overfitting:       BAJO ✅
        ├─ Train: 99.60%
        ├─ Val:   92.89%
        └─ Test:  90.51%
   (Generaliza bien)
```

**En producción**: De 1000 residuos, ~905 se clasifican automáticamente, ~95 necesitan revisión humana.

---

## SLIDE 6: NUEVA EVALUACIÓN DE ROBUSTEZ (MULTI-SEMILLA)

Se añadieron dos notebooks específicos:

- `notebooks/Evaluacion_modelo8.ipynb`
- `notebooks/Evaluacion_modelo9.ipynb`

Semillas evaluadas: **42, 123, 7, 99**

**Resultados promedio**:

| Modelo | Train | Val | Test |
|--------|-------|-----|------|
| EfficientNetB0 | 99.85% | 91.01% | 87.65% |
| DenseNet121 | 99.90% | 89.23% | 88.83% |

✅ Conclusión de robustez: **Modelo 9 mantiene mejor media en test**.

---

## SLIDE 7: CONCLUSIONES Y FUTURO

### ✅ Logramos
- Clasificador funcional de **90.51% de accuracy**
- Validación de robustez con múltiples semillas
- Desde scratch hasta transfer learning
- Sistema reproducible y escalable

### 🚀 Próximo nivel
- ⬆️ Más imágenes (data augmentation)
- ⬆️ Modelos más nuevos (Vision Transformers)
- ⬆️ Ensembles (combinar múltiples redes)
- ➡️ Deployment en dispositivos móviles

---

## SLIDE 8: IMPACTO

💡 **Este sistema puede**:
- Automatizar plantas de reciclaje
- Reducir costos operacionales
- Mejorar tasas de recuperación
- Sumarse a soluciones sostenibles

**Status**: Viable para automatización asistida (requiere supervisión humana del 9.49% de casos dudosos)

---

## PREGUNTA FINAL

> ### ¿Preguntas?
> 
> 📊 Resultados completos en `INFORME.md`  
> 💻 Código reproducible en `/src`, `/notebooks`, `Evaluacion_modelo8.ipynb` y `Evaluacion_modelo9.ipynb`

---

**Tiempo transcurrido: ~5 minutos**  
*Gracias por su atención* 👋
