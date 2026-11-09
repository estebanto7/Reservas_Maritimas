# ¿Cuál es el objetivo general de la actividad?

Consiste en analizar el comportamiento de un mercado de seguros de embarcaciones y calcular las reservas que necesita una aseguradora para cumplir con sus obligaciones*

Se puede dividir en 3 grandes partes:

## Análisis del mercado (1, 2, 3)
Estudiar las primas cobradas, los siniestros y su comportamiento a través de los años

## Siniestralidad y Conglomerados (4)
Medir que tan grandes son los siniestros respecto a las primas y agrupar tipos de embarcaciones con comportamientos similares

## Calculo de Reservas (5, 6) 
Estimar cuánto dinero se necesita reservar para riesgos aún vigentes y siniestros ocurridos todavía no reportados


Datos de primas y siniestros (por año y tipo de embarcación)
 ---> Calcular factores de siniestralidad anual (Siniestros totales / primas emitidas)
 ---> Análisis de conglomerados (Agrupar tipos de embarcación con factores anuales semejantes, por ejemplo mediante Ward con distancia euclidiana)
---> Chain Ladder, por separado (Construir triángulos de desarrollo y proyectar siniestros para estimar la reserva solicitada)


# ¿Qué se pide en cada apartado?

## 1. Actualización de cifras
**Se busca:** que la inflación no distorsione las comparaciones entre años

- LLevar los montos monetarios de distintos años a una base comparable, indicando el factor de actualización y su fiente, por ejemplo, mediante el INPC **(Crear una función para convertir una base de datos a cifras comparables)**


## 2. Análisis de la prima emitida
**Se busca**: entender cómo se comporta el volumen de primas del mercado y como esta distribuido

- a) Ver como aumenta o disminuye la prima emitida total por año **(Line chart)**
- b) Comparar las comisiones directas con las primas emitidas  
- c) Comparar las primas retenidas con las emitidas
- d) Analizar la distribución de primas por tipo de embarcación, identificando cuáles aprotan más o menos y cómo cambian con el tiempo

## 3. Análisis del monto de siniestros
**Se busca**: entender cúando ocurren mayores périddas y cuáles son sus principales causas

- a) Calcular cuánto suman los siniestros de cada año y estudiar su tendencia
- b) Identificar las principales causas de siniestros y cuánto representan del total anual 

## 4. Factores de siniestralidad y conglomerados
**Se busca:** reconocer qué segmentos presentan siniestralidad alta, baja o comportamientos similares 

$$ Siniestralidad = \frac{Siniestros totales}{Prima emitida}$$

- a) Calcular la siniestralidad anual por tipo de embarcaci[on y para todo el mercado
- b) Comparar estos factores entre años, identificando máximos, mínimos, valores atípicos y cambios
- Agrupar los tipos de embarcaciones según sus factores anuales de siniestraldiad **(Análisis 
de Conglomerados)**

## 5. Reserva de Riesgos en Curso (RRC)
**Se busca:** estimar la reserva necesaria para cubrir obligaciones futuras por pólizas que siguen vigentes, comparando los resultados de los tres métodos

- Calcular al 31 de diciembre de 2025 cuánto dinero corresponde a riesgos que todavía estan cubiertos por pólizas vigentes.
- Se comparan tres métodos: exacto, venticuatroavos y estatutario

## 6. Reserva de siniestros ocurridos no reportados (SONR)
**Se busca**: estimar estas obligaciones pendientes con diferentes metodologías y explicar las diferencias 

- Calcular al 31 de diciembre de 2025 la reserva correspondiente a siniestros que ya ocurrieron, pero todavía no se reportaron.
- Se compara Chain Ladder, método de la razón y método estatutario

## 7. Conclusiones 
**Se busca:** interpretar qué aprendiste del comportamiento del mercado, sus riesgos, su segmentación y sus reservas

Explicar los prinicpales resultados de cada uno de los seis apartados anteriores. 


# Metodología del análisis de conglomerados

**Objetivo:** Identificar grupos homogéneos de tipos de embarcación según su comportamiento histórico de siniestralidad, para apoyar la estimación de reservas de riesgos en curso.

**Periodo:** 2015–2022 (ocho años de origen).  
**Unidad de análisis:** Tipo de embarcación.

## 3.1. Preparación y cálculo de factores de siniestralidad

1. **3.1.1. Siniestralidad del mercado:** analizar la evolución del factor de siniestralidad anual.
2. **3.1.2. Siniestralidad por tipo:** calcular los factores anuales de cada tipo de embarcación.
3. **3.1.3. Selección de tipos:** conservar aquellos con primas positivas durante los ocho años, que representan el 96.29% de las primas emitidas del periodo.
4. **3.1.4. Matriz de análisis:** construir una matriz de tipos de embarcación × ocho factores anuales.

## 3.2. Construcción y preparación de características

1. **3.2.1. Formulación del problema:** definir el objetivo de agrupación y las unidades de análisis.
2. **3.2.2. Construcción de características:**
   - Siniestralidad acumulada: siniestros totales / primas totales.
   - Variabilidad temporal: dispersión de los factores anuales.
   - Recurrencia: proporción de años con siniestralidad superior a 1.
   - Tendencia: pendiente de la siniestralidad a lo largo del tiempo.
   - Volumen de primas: indicador auxiliar de experiencia.
3. **3.2.3. Diagnóstico multivariado:** estudiar distribuciones, valores extremos y correlaciones de Pearson y Spearman entre características.
4. **3.2.4. Preprocesamiento:** evaluar transformaciones y estandarizar las variables seleccionadas.

## 3.3. Análisis de conglomerados

1. Aplicar Ward, Complete Linkage, Average Linkage y K-means.
2. Evaluar distintos números de grupos (\(k\)).
3. Comparar Silhouette, Calinski-Harabasz y tamaños de los grupos.
4. Seleccionar agrupaciones considerando homogeneidad e interpretación.

## 3.4. Robustez de los conglomerados

1. Identificar la influencia de observaciones extremas.
3. Evaluar sensibilidad con y sin observaciones extremas, sin eliminarlas definitivamente.
4. Comparar los cambios en la composición de los conglomerados.

## 3.5. Estabilidad temporal

1. Construir ventanas móviles de cuatro años (2015–2018 hasta 2019–2022).
2. Comparar agrupaciones entre ventanas consecutivas mediante ARI.
3. Comparar los periodos independientes 2015–2018 y 2019–2022.
4. Identificar tipos con agrupaciones persistentes o cambiantes.

## 3.6. Interpretación y evaluación actuarial

1. Caracterizar los conglomerados según su siniestralidad y variabilidad.
2. Evaluar homogeneidad, estabilidad y volumen de experiencia.
3. Examinar la viabilidad de aplicar credibilidad actuarial.
4. Verificar la correspondencia temporal entre primas y siniestros.
5. Determinar la utilidad y limitaciones de los conglomerados para estimar reservas de riesgos en curso.
