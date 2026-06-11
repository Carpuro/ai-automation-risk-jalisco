# Capítulo 3. Datos y métodos

Este capítulo documenta los datos y la maquinaria empírica de la tesis. La
exposición sigue el orden de la cadena causal del capítulo anterior (§2.7):
primero las fuentes y la arquitectura de datos (3.1); después la
construcción y validación de los dos índices de exposición (3.2, 3.3) y la
cadena de cruce que los conecta con el mercado laboral mexicano (3.4); en
seguida la arquitectura de dos niveles que organiza los modelos (3.5); y por
último la estrategia de inferencia y robustez (3.6). Todos los resultados de
la tesis son reproducibles: cada tabla y figura es producto de un script
versionado, y el repositorio documenta el orden completo de reconstrucción
desde cero.

## 3.1 Fuentes y arquitectura de datos

La tesis integra catorce fuentes en una base de datos SQL Server
(~45 tablas) que funciona como fuente única de verdad; ningún resultado se
calcula sobre archivos sueltos. Las fuentes se agrupan según el eslabón de
la cadena causal que alimentan:

**Contenido ocupacional y exposición (eslabones 1–2).** La base O\*NET 28.3
aporta los descriptores de habilidades, actividades y contextos de trabajo
(escalas de importancia, nivel y contexto). El apéndice de datos de Felten,
Raj y Seamans (2021) aporta la matriz de afinidad habilidad×aplicación y el
índice AIOE publicado, usado como patrón de validación. La base de
capacidades de Epoch AI aporta los resultados de la frontera de modelos de
lenguaje en doce benchmarks públicos (2022–2026). El archivo comparativo de
índices del proyecto Moravec aporta, al grano ocupacional, los índices
externos de contraste: exposición robótica, a software y a IA por patentes
(Webb, 2020), Frey-Osborne, aptitud para aprendizaje automático (SML),
exposición a GPT (Eloundou et al., 2023) e intensidades de rutina. El Índice
Económico de Anthropic aporta el uso observado de IA por ocupación (variable
dependiente del Nivel 1) y, en su entrega por países, el perfil ocupacional
de uso de México empleado para contrastar el supuesto de trasplante (§2.6).

**Mercado laboral de Jalisco (eslabón 3).** La ENOE del tercer trimestre de
2024 aporta el microdato de trabajadores: 13,839 personas en la muestra de
Jalisco, de las cuales 6,147 ocupadas con ocupación SINCO a cuatro dígitos.
Este grano fino no está disponible directamente: el cuestionario
sociodemográfico (SDEMT) no registra la ocupación detallada, que se recupera
de la pregunta 3 del cuestionario de ocupación y empleo (COE1) mediante el
empate por llave de persona (entidad, control, UPM, vivienda seleccionada,
hogar, renglón y periodo). Para el análisis de empleo total con
informalidad, se procesaron además los microdatos de once trimestres
consecutivos (2022-T1 a 2024-T3), clasificando cada ocupado con la variable
oficial de informalidad del empleo principal. Para el contraste nacional, la
misma cadena se replicó sobre el microdato completo (192,961 ocupados con
ocupación detallada; 59.5 millones ponderados).

**Incentivo económico y sustitución realizada (eslabones 4–5).** Los Censos
Económicos del INEGI aportan, para Jalisco, un panel balanceado de 19
sectores SCIAN en cinco levantamientos (2003, 2008, 2013, 2018, 2023) con
personal ocupado, remuneraciones, contribuciones sociales, acervo de activos
fijos, depreciación, acervo de equipo de cómputo y valor agregado: el insumo
del IRA y del contraste de H4. El registro del IMSS (vía IIEG) aporta el
empleo formal mensual por sector, 2000–2024. La Secretaría de Economía
aporta los flujos trimestrales de inversión extranjera directa por entidad y
tipo (2006–2023). La Federación Internacional de Robótica (IFR) aporta las
instalaciones anuales de robots industriales en México (2019–2024,
extraídas de los resúmenes ejecutivos públicos) y, vía Our World in Data, el
acervo operativo mundial (2012–2024), insumo de la curva r(t).

**Incidencia y percepción (eslabón 6).** El Latinobarómetro aporta cuatro
oleadas mexicanas (2017, 2018, 2020, 2023) del ítem de expectativa de
desplazamiento por robots/IA, con escalas verificadas contra las etiquetas
originales de cada archivo. Los Censos Económicos en su grano municipal y la
cartografía del INEGI alimentan el mapa de los 125 municipios.

**Cruces de clasificación.** El puente entre la clasificación mexicana de
ocupaciones (SINCO 2011) y la estadounidense (SOC) proviene de las tablas
comparativas oficiales del INEGI; el puente ISCO↔SOC, del crosswalk de ESCO.

## 3.2 El índice cognitivo: DBOE

El DBOE (*Dynamic Benchmark-based Occupational Exposure*) extiende el AIOE
de Felten et al. (2021) con una dimensión temporal medida. Su construcción
tiene tres componentes.

**Pesos ocupacionales.** Para cada ocupación *o* y habilidad *k*, el peso
W_ok suma la importancia y el nivel de la habilidad, cada uno estandarizado
entre ocupaciones: W_ok = z_occ(importancia_ok) + z_occ(nivel_ok). Los
códigos O\*NET de ocho dígitos se agregan a su ocupación SOC de seis
dígitos.

**Capacidad por aplicación en el tiempo.** De las diez aplicaciones de IA de
la matriz de Felten, se retienen las tres que los modelos de lenguaje
efectivamente ejecutan: modelado de lenguaje, comprensión lectora y juegos
abstractos de estrategia (esta última como proxy de razonamiento,
operacionalizada con benchmarks de matemática de competencia y ajedrez; la
elección se discute y se somete a robustez en §3.6). Para cada aplicación
*j* y año *t*, la capacidad c_j(t) es el promedio, entre sus benchmarks, del
máximo histórico alcanzado por algún modelo publicado hasta el cierre de
*t*. Al ser un máximo acumulado, la frontera es monótona por construcción;
la convención «el benchmark aún no existe ⇒ frontera 0» se aplica benchmark
por benchmark antes de promediar, lo que evita que la entrada tardía de
benchmarks difíciles deprima artificialmente la curva agregada.

**Agregación.** La exposición de la habilidad *k* en el año *t* es
A_k(t) = Σ_j rel[k,j]·c_j(t), con rel la matriz de afinidad de Felten; el
índice es DBOE_o(t) = Σ_k z_hab(A_k(t))·W_ok, estandarizando A_k entre
habilidades. La doble estandarización no es opcional: omitirla invierte el
ordenamiento, porque el piso de afinidad de la matriz hace dominar a las
ocupaciones con muchas habilidades de alto nivel.

**Validación.** Reconstruido con las diez aplicaciones y capacidad unitaria,
el procedimiento reproduce el AIOE publicado con r = 0.942 (N = 681); la
versión dinámica de 2026 sigue al AIOE específico de modelado de lenguaje
con r = 0.925. Una propiedad medida del índice se reporta como limitación y
como hallazgo: el reordenamiento ocupacional entre 2022 y 2026 es mínimo
(Spearman 0.998); la señal temporal del DBOE reside en el **nivel** de la
curva de capacidad —el razonamiento pasa de 0.00 a 0.79— y no en el
reordenamiento, de modo que el corte transversal (estandarizado, dboe_z)
actúa como variable de exposición y la curva c(t) como motor de la
proyección.

## 3.3 El índice encarnado: DEOE

El DEOE (*Dynamic Embodied Occupational Exposure*) es el espejo físico del
DBOE, construido con la misma lógica de ponderación. Quince descriptores de
O\*NET —seis actividades de trabajo en escala importancia+nivel y nueve
contextos de trabajo en escala de frecuencia— se agrupan en cinco
subdominios teóricamente fundados: trabajo físico general y manejo de
objetos (α = 0.95); operación y reparación de maquinaria (α = 0.82);
vehículos y trabajo de campo (α = 0.91); rutina física (α = 0.57, el
subdominio débil, dos ítems); y destreza manual en espacios restringidos
(α = 0.75). El índice resumen es el primer componente principal de los cinco
subdominios (63% de la varianza, todas las cargas positivas).

Dos decisiones de construcción merecen registro. Primera: el resumen es un
componente principal y no una suma con signos impuestos, porque la
validación externa contradijo la teoría previa —la destreza manual,
postulada por Frey y Osborne como cuello de botella protector, correlaciona
+0.74 con la exposición robótica real medida por patentes: la frontera
robótica ya erosionó ese cuello de botella, y los datos, no el supuesto,
fijan los signos. Segunda: la validación es convergente y discriminante a la
vez. El DEOE correlaciona +0.76 con el índice robótico de Webb y +0.82 con
la intensidad de rutina manual (convergencia), y ≈0 con los índices
cognitivos (discriminación); la imagen especular del DBOE, que correlaciona
−0.74 con el índice robótico. En el proceso se documentó que dos índices
publicados como físicos (Moravec, factibilidad RL) son empíricamente
cognitivos (r = −0.66 contra el contenido físico real), lo que motivó la
construcción de un eje encarnado propio.

**La capa dinámica.** No existe para la robótica una batería de benchmarks
equivalente a la de los modelos de lenguaje; la curva temporal del eje
encarnado se construye, siguiendo el estándar de la literatura
(Acemoglu y Restrepo, 2020; Graetz y Michaels, 2018), sobre **adopción**: el
acervo operativo mundial de robots industriales (IFR), indexado al año base,
con proyección 2025–2030 en tres escenarios derivados de los datos (TCAC
trianual observada de 10.3% como base; la mitad como conservador; 1.5 veces
como acelerado, justificado por el canal de nearshoring de §2.6). Las
instalaciones anuales en México anclan el nivel local. La asimetría entre
una curva de capacidad (cognitiva) y una de adopción (encarnada) se declara
explícitamente como supuesto del marco.

## 3.4 La cadena de cruce y su cobertura

Conectar exposición (grano SOC) con trabajadores (grano SINCO) exige un
puente de clasificaciones. La tesis emplea las tablas comparativas oficiales
del INEGI (SINCO 2011 ↔ SOC), que mapean 403 ocupaciones SINCO de cuatro
dígitos a códigos SOC de grupo (726 relaciones uno-a-varios). Dado que el
cruce llega al SOC *broad* y la exposición reside en el SOC detallado, la
exposición se agrega primero al grupo (promedio) y se une por esa llave. La
cadena validada —trabajador ENOE → SINCO-4d → SOC broad → exposición— cubre
el 88% del empleo ponderado de Jalisco (y entre 81% y 92% en los 32
estados). El 12% restante no se ignora: la sección 3.6 describe el análisis
de cotas que delimita su efecto posible, y el diagnóstico muestra que tres
cuartas partes de ese peso corresponden a códigos de la revisión SINCO 2019
ausentes de las tablas comparativas de 2011, concentrados en divisiones
manuales.

## 3.5 La arquitectura de dos niveles

**Nivel 1: el modelo ocupacional (N = 678 SOC).** La pregunta es si los
índices propios explican el uso observado de IA más allá de la línea base de
Frey-Osborne. La variable dependiente —uso observado por ocupación, Índice
Económico de Anthropic— está inflada en ceros (52% de las ocupaciones sin
uso registrado; mediana 0), de modo que una regresión única estaría mal
especificada. Se estima un **modelo de dos partes (hurdle)**: un logit para
el margen extensivo (¿la ocupación registra uso alguno?) y una regresión
sobre el logaritmo del uso condicional a uso positivo (margen intensivo;
N = 326). Cada parte se estima jerárquicamente —Frey-Osborne, +DBOE,
+DEOE— con pruebas de razón de verosimilitud y de F incremental, y los
índices propios se comparan además contra los rivales publicados (AIOE, SML,
Eloundou) como predictores individuales. La validez del trasplante del
objetivo (uso estadounidense) se contrasta con el perfil mexicano de la
entrega por países del propio índice.

**El contraste de H4: el panel de sustitución realizada.** La moderación
económica no se asume: se contrasta contra veinte años de comportamiento
revelado. Sobre el panel balanceado de los Censos (19 sectores × 4
transiciones censales, N = 76) se regresan los resultados de sustitución de
cada transición —cambio logarítmico del capital por trabajador, del equipo
de cómputo por trabajador y del empleo, y cambio de la participación laboral
en el valor agregado— sobre el **IRA rezagado** (el incentivo observado al
inicio de la transición), con efectos fijos de periodo y errores agrupados
por sector. La identificación es transversal dentro de periodo; los efectos
fijos absorben la inflación y los choques comunes. El estadístico de interés
es el coeficiente del incentivo rezagado; su inferencia exacta y su placebo
de temporalidad inversa se describen en §3.6.

**El experimento natural y la absorción.** El efecto temprano del choque de
IA generativa se examina con diferencias en tendencias alrededor de
noviembre de 2022: log del empleo sobre efectos fijos de sector y de mes,
con interacciones post×DBOE y post×DEOE, primero sobre el empleo formal
mensual del IMSS (8 sectores, ventana principal 2021–2024) y después —para
responder la objeción de informalidad— sobre el **empleo total** trimestral
de la ENOE (formal más informal, 19 sectores, 2022-T1 a 2024-T3), añadiendo
como segunda variable dependiente la participación informal del sector (la
«firma de absorción»: si el desplazamiento formal se recoloca en la
informalidad, esa participación debe crecer en los sectores expuestos).

**Complementos del lado de la demanda.** Una ecuación de Mincer ponderada
(log del salario por hora sobre escolaridad, edad, sexo, formalidad y
ruralidad) estima si el trabajo expuesto ya paga una prima —el costo laboral
que el adoptante se ahorra—, con interacción exposición×educación superior.
El registro de tareas emergentes de O\*NET alimenta el contraste descriptivo
de reinstalación. Los flujos de IED por entidad y tipo documentan el canal
de nearshoring.

**Nivel 2: proyección de escenarios 2025–2030.** La proyección es análisis
de escenarios anclado en datos —no un pronosticador entrenado: no existe
panel de automatización observada—. La presión sobre el sector *s* en el año
*t* y el escenario *k* se define como
presión(s,t,k) = pct(exposición_s) × curva_k(t) × pct(IRA_s), donde pct(·)
es el rango percentil entre sectores, las curvas c(t) y r(t) se re-basan en
2024 = 1.00 (el último año observado de ambas, de modo que la presión se lee
como crecimiento desde el presente) y la moderación por IRA aplica la
relación evidenciada por H4. En el nivel del trabajador, la cuantificación
de población en riesgo fija el umbral τ en el tercil superior (ponderado por
empleo) de la presión total de 2024 y cuenta, por escenario, a los
trabajadores cuya presión de 2030 supera esa barra de hoy; el perfil de esa
población (educación, sexo, formalidad, polo dominante) y su multiplicador
de severidad por informalidad completan la incidencia.

## 3.6 Inferencia, robustez y reproducibilidad

Los contrastes sectoriales de esta tesis operan con pocas unidades
transversales (8 a 19 sectores), donde los errores estándar asintóticos son
optimistas. Por ello, todos los coeficientes centrales se acompañan de
**inferencia por permutación** (B = 1,000): el tratamiento a nivel de sector
—los puntajes de exposición; la trayectoria completa del IRA— se reasigna
aleatoriamente entre sectores conservando su estructura temporal, el modelo
se reestima y el valor p empírico es la fracción de permutaciones con
coeficiente igual o mayor en valor absoluto. Este criterio es exacto bajo la
hipótesis nula y válido con cualquier N. Su aplicación tuvo consecuencias en
ambas direcciones, que el capítulo 5 reporta: confirmó con holgura el
contraste de H4 y degradó a no significativo un coeficiente del experimento
natural que la inferencia asintótica declaraba significativo.

La batería de robustez incluye además: (i) **cotas de cobertura** para el
12% de trabajadores sin exposición asignada —se recalculan las medias de
Jalisco asignando a todos los no cubiertos el mínimo y después el máximo
observado de cada eje; (ii) **sensibilidad de umbral** para la cuantificación
de población en riesgo (mediana, tercil, cuartil y decil), donde lo que debe
sobrevivir es el orden de escenarios, no la cuenta absoluta; (iii)
**exclusión de benchmarks uno a uno** en el DBOE, dirigida a su elección más
discutible (ajedrez como proxy de estrategia abstracta); (iv) **placebo de
temporalidad inversa** para H4 (¿"predice" el incentivo la profundización del
periodo anterior?); y (v) análisis de sensibilidad del componente principal
del DEOE excluyendo su subdominio débil.

Por último, la reproducibilidad es una propiedad del diseño y no un anexo:
cada tabla y figura de los capítulos 4 a 6 es la salida de un script
identificado, la base de datos puede reconstruirse desde cero con el DDL y
el orden de ejecución documentados en el repositorio, y los números del
texto se extraen de las tablas de resultados, no se transcriben a mano.
