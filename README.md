# tennis_prediction
Este repositorio contiene un pipeline completo de Machine Learning (*End-to-End*) diseñado para predecir el resultado de partidos de tenis profesionales utilizando un histórico de datos masivo (2000-2024).

El proyecto destaca por el uso de técnicas de **ingeniería de características temporales**, **calibración probabilística** y **explicabilidad del modelo (XAI)**.

## 📌 Contexto y Reto de Negocio
En el tenis, las métricas estáticas (como el ranking actual) suelen ser insuficientes para capturar la realidad competitiva. Este proyecto resuelve el problema mediante la creación de variables dinámicas que miden la "forma" real y el nivel relativo (Elo) de los jugadores en el momento exacto del enfrentamiento. 

Se han desarrollado dos modelos específicos para capturar las naturalezas distintas de los circuitos:
1.  **Modelo ATP:** Enfocado en la estabilidad y especialización de la élite.
2.  **Modelo Challenger:** Adaptado a la alta varianza y al desarrollo de jóvenes talentos.

## 🤖 Metodología de Desarrollo: AI-Augmented Engineering
Este proyecto ha sido desarrollado mediante una colaboración estratégica con **Inteligencia Artificial**, permitiendo un ciclo de desarrollo ágil:
* **Rol del Data Scientist:** Arquitectura del sistema, diseño de hipótesis (como el uso de Brier Score), validación de la simetría de datos y supervisión técnica.
* **Rol de la IA:** Aceleración de la escritura de código, optimización de algoritmos de limpieza y generación de estructuras de modelado.

## 🚀 Pipeline Técnico

### 1. Data Engineering & Feature Engineering (`preparacion datos_ganador.ipynb`)
* **Simetrización del Dataset:** Duplicación de datos invirtiendo objetivos y atributos para eliminar el sesgo de "Jugador A" y obligar al modelo a aprender la dinámica del H2H.
* **Optimización de Ventana Temporal:** Uso de *Rolling Windows* de **10 partidos** para capturar la racha actual sin introducir ruido histórico.
* **Sistema ELO Dinámico:** Cálculo de `Ratio_ELO_zscore` específico por superficie.
* **Normalización (Z-Scores):** Estandarización de métricas (Aces, Dobles faltas, etc.) para permitir comparaciones entre distintas eras y superficies.

### 2. Modelado Predictivo (`modelos_predicción_...ipynb`)
* **Algoritmo:** **XGBoost Classifier** con regularización avanzada (`reg_alpha`, `reg_lambda`, `gamma`) para evitar el sobreajuste.
* **Validación Temporal (`TimeSeriesSplit`):** Entrenamiento (años 2000-2023), Validación (año 2024) y Test Final (año 2025).
* **Evaluación Probabilística:** Optimización basada en el **Brier Score** (~0.219) y **Log-Loss**, garantizando que las probabilidades de victoria estén matemáticamente calibradas.

## 📊 Explicabilidad del Modelo (SHAP Analysis)

El uso de **SHAP** revela que la IA utiliza estrategias distintas según el circuito, validando el conocimiento de negocio:

| Circuito | Driver Principal (SHAP) | Lógica de Negocio Detectada |
| :--- | :--- | :--- |
| **ATP** | `elo_ratio_surface` | En la élite, el rendimiento histórico específico en la superficie es el predictor más fiable. |
| **Challenger** | `rank_ratio` / `age_diff` | En el circuito de ascenso, la brecha de ranking y la juventud (energía física) son determinantes ante la irregularidad del Elo. |

*Los gráficos SHAP muestran que el modelo penaliza la edad avanzada y premia la especialización técnica en el circuito principal.*

## 📈 Resultados Finales

Los modelos fueron evaluados con un *dataset* completamente aislado (partidos de 2025) para garantizar un escenario de predicción real libre de sesgos y sobreajuste:

**🎾 Modelo ATP (Circuito Principal)**
* **Accuracy:** 64.22%
* **Brier Score:** 0.219 (Excelente calibración en un entorno de alta jerarquía y estabilidad).

**🎾 Modelo Challenger (Circuito de Ascenso)**
* **Accuracy:** 63.53%
* **Brier Score:** 0.228 (Resultado muy sólido teniendo en cuenta la alta volatilidad e irregularidad intrínseca de este circuito).

**Rendimiento General:** Ambos modelos mantienen su capacidad predictiva en datos no empleados en el entrenamiento (año 2025), demostrando robustez ante el cambio de tendencia, relevo generacional y nuevas dinámicas en el circuito profesional.

## 🛠️ Stack Tecnológico
* **Lenguaje:** Python
* **Librerías Clave:** `XGBoost`, `Scikit-Learn`, `SHAP`, `Pandas`, `Matplotlib`.
* **Entorno:** Jupyter Notebooks / Google Colab.

---
**👤 Diego Herrera Ochoa**
*Data Scientist | PhD en Ciencias de la Salud*
[LinkedIn](https://www.linkedin.com/in/diego-herrera-ochoa-314015377)
