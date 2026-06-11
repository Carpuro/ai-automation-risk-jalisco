# Capítulo 2. Marco teórico

Este capítulo construye la base conceptual de la tesis. Su argumento puede
resumirse en una frase: el riesgo de automatización por inteligencia
artificial no es un fenómeno único sino la intersección de **dos fronteras
tecnológicas distintas** con **un mercado que decide adoptar solo cuando es
rentable**, y ambas cosas —la exposición técnica y la decisión económica—
son medibles. Las secciones siguientes desarrollan los seis bloques teóricos
que sostienen ese argumento, y la sección final deriva de ellos las cuatro
hipótesis que el resto de la tesis somete a prueba.

## 2.1 El marco de tareas: la tecnología sustituye tareas, no empleos

La economía moderna de la automatización descansa sobre el modelo de tareas
introducido por Autor, Levy y Murnane (2003) y generalizado por Acemoglu y
Autor (2011). Su premisa central invierte la intuición popular: la unidad
relevante del análisis no es el empleo sino la **tarea**. Una ocupación es un
paquete de tareas heterogéneas —cognitivas y manuales, rutinarias y no
rutinarias— y la tecnología no compite contra "el contador" o "el soldador"
sino contra cada una de las tareas que componen su jornada.

De esta premisa se siguen dos consecuencias que organizan el diseño completo
de esta tesis. La primera es metodológica: si la tecnología sustituye tareas,
la exposición a la automatización debe medirse a partir del **contenido de
tareas y habilidades** de cada ocupación, no de su título. Esto exige una
fuente que describa sistemáticamente qué hacen las ocupaciones; el estándar
internacional es la base O\*NET del Departamento del Trabajo de Estados
Unidos, que descompone cada ocupación en habilidades, actividades y contextos
de trabajo con escalas de importancia y nivel. Toda la familia de índices de
exposición que la literatura ha producido —y los dos índices propios de esta
tesis— se construye sobre ese sustrato (la discusión sobre la portabilidad de
ese contenido a México se desarrolla en la sección 2.6).

La segunda consecuencia es interpretativa: el "riesgo de automatización" no
equivale a destrucción binaria de empleos. Una ocupación puede ver
automatizada una fracción de su paquete de tareas con efectos sobre salarios,
composición del trabajo y contratación **antes** de que se observe efecto
alguno sobre el número de empleos. Esta observación, aparentemente menor,
restringe qué tipo de evidencia puede esperarse en las fases tempranas de una
tecnología —un punto al que la sección 2.5 regresa y que resulta decisivo
para interpretar los resultados del capítulo 5.

## 2.2 Desplazamiento, reinstalación y la condición de rentabilidad

Si el marco de tareas dice *dónde* puede actuar la tecnología, el modelo de
Acemoglu y Restrepo (2018, 2019) dice *qué pasa después*. La automatización
es una **carrera entre dos fuerzas**: el efecto desplazamiento —las máquinas
toman tareas que realizaban personas— y el efecto reinstalación —el cambio
técnico crea tareas nuevas en las que el trabajo humano tiene ventaja
comparativa. El efecto neto sobre el empleo no está determinado por la
teoría: es una pregunta empírica, y una tesis creíble está obligada a buscar
evidencia de ambos lados de la carrera.

Tres elementos de este modelo son constitutivos del diseño de esta
investigación. El primero es la distinción entre **factibilidad y adopción**.
Que una tarea sea técnicamente automatizable no implica que se automatice: la
sustitución ocurre cuando el costo del capital que ejecuta la tarea es menor
que el costo del trabajador que la realiza. La adopción depende del cociente
entre el salario y el costo del capital, no de la capacidad técnica en
abstracto. Esta es la definición teórica del Índice de Rentabilidad de la
Automatización (IRA) que esta tesis construye a partir de los Censos
Económicos —costo laboral sobre proxies del costo de capital— y la razón de
ser de la hipótesis H4: la exposición técnica solo se convierte en presión
real de automatización donde el incentivo económico existe.

El segundo elemento es el concepto de **tecnologías mediocres** ("so-so
technologies", Acemoglu y Restrepo, 2019): automatización lo bastante
rentable para desplazar trabajo pero no lo bastante productiva para expandir
la demanda del sector. Es el peor escenario distributivo, y su firma empírica
no son los despidos masivos sino el **crecimiento perdido**: sectores que
dejan de crear los empleos que habrían creado. Como mostrará el capítulo 5,
esa es exactamente la forma que la sustitución ha tomado en Jalisco durante
las últimas dos décadas.

El tercer elemento es el lado de la reinstalación. La literatura empírica
sobre exposición rara vez lo mide; esta tesis lo aborda con el registro de
tareas emergentes de O\*NET —ocupaciones que ganan tareas nuevas— como
indicador, imperfecto pero observable, de la capacidad de renovación de cada
ocupación.

## 2.3 Dos fronteras tecnológicas: la paradoja de Moravec como teoría de los dos ejes

La decisión de diseño más importante de esta tesis —medir la exposición en
dos ejes separados, cognitivo y encarnado— no es una preferencia taxonómica:
es una predicción teórica con raíz en la paradoja de Moravec (Moravec, 1988).
La observación es clásica en inteligencia artificial: el razonamiento
abstracto de alto nivel resulta computacionalmente barato, mientras que la
percepción y la destreza sensoriomotora —triviales para cualquier humano—
resultan extraordinariamente costosas. "La inteligencia artificial" no es,
por tanto, una tecnología sino **dos fronteras con física distinta, curvas de
costo distintas y adoptantes distintos**:

| | Frontera cognitiva (LLMs) | Frontera encarnada (robots) |
|---|---|---|
| Motor de capacidad | desempeño de modelos de lenguaje y razonamiento | manipulación, navegación, costo por robot |
| Capital requerido | casi nulo (un teléfono, una suscripción) | capital fijo formal (una empresa compra la máquina) |
| Canal de difusión | persona por persona, instantáneo, cruza la informalidad | inversión por inversión, lento, solo sector formal |
| Curva medida en esta tesis | c_j(t), frontera de benchmarks | r(t), acervo mundial operativo de robots (IFR) |

De esta distinción se derivan dos predicciones no obvias. La primera: la
exposición ocupacional a ambas fronteras debe estar **negativamente
relacionada**. Las habilidades baratas para el software son caras para los
robots y viceversa; las ocupaciones más expuestas a los modelos de lenguaje
deberían ser las menos expuestas a la robótica. Webb (2020) documentó esta
oposición con texto de patentes; esta tesis la formula como hipótesis H2 y la
somete a prueba con índices propios mediante análisis factorial exploratorio
y confirmatorio (capítulo 4).

La segunda predicción es una exigencia de higiene metodológica: **todo índice
de exposición debe declarar su frontera**. Un índice que se presenta como
medida de automatización física pero correlaciona con el contenido cognitivo
del trabajo está midiendo la frontera equivocada. El hallazgo de auditoría de
esta tesis —dos índices publicados como físicos (Moravec auto_w y la
factibilidad por aprendizaje por refuerzo) resultan empíricamente cognitivos,
con correlación negativa de −0.66 contra el contenido físico real del
trabajo— ilustra el costo de omitir esa verificación, y explica por qué fue
necesario construir un índice encarnado propio.

La asimetría de capital de la tabla anterior tiene además una implicación
distributiva central para una economía como la mexicana: la frontera
encarnada solo puede llegar a través de la inversión formal, mientras que la
frontera cognitiva se difunde por teléfono y alcanza también al sector
informal. Los instrumentos estadísticos formales (censos, registros del
IMSS) miden bien el canal robótico y mal el canal cognitivo —una asimetría
que el diseño empírico de los capítulos 5 y 6 incorpora de manera explícita.

## 2.4 La medición de la exposición: linaje y posición de esta tesis

La literatura de medición puede leerse como una sucesión de respuestas a una
misma pregunta: ¿cómo sabemos qué ocupaciones toca la tecnología?

**Frey y Osborne (2017)** inauguraron el campo: un taller de expertos
etiquetó 70 ocupaciones como automatizables o no, y un clasificador de
procesos gaussianos extrapoló esas etiquetas a 702 ocupaciones a través de
nueve "cuellos de botella de ingeniería". Su célebre 47% de empleo en riesgo
fijó la agenda pública, pero el método recibió dos críticas duraderas: opera
sobre ocupaciones completas en lugar de tareas —Arntz, Gregory y Zierahn
(2016) muestran que el enfoque de tareas reduce el estimado a una fracción— y
sus anclas son juicios subjetivos de un grupo pequeño.

**Webb (2020)** sustituyó el juicio experto por texto: la superposición entre
el lenguaje de las patentes y el lenguaje de las descripciones de tareas
produce índices separados para software, robots e inteligencia artificial, y
con ellos la primera evidencia clara de que las fronteras apuntan a
ocupaciones diferentes.

**Felten, Raj y Seamans (2021)** sistematizaron la medición con el AIOE:
52 habilidades de O\*NET cruzadas con 10 familias de aplicaciones de IA,
ponderadas por la importancia de cada habilidad en cada ocupación. El
resultado es reproducible y transparente, pero **estático**: la capacidad de
la tecnología entra como una constante, como si la frontera de 2021 fuera la
de siempre.

**Eloundou, Manning, Mishkin y Rock (2023)** llevaron la medición al nivel de
tarea usando al propio modelo —GPT-4 califica qué tareas puede realizar— con
alto poder predictivo pero con la subjetividad trasladada del experto humano
al modelo, y de nuevo como fotografía de un instante.

**El Índice Económico de Anthropic (2025–2026)** añadió lo que faltaba en
toda la cadena: comportamiento observado. Millones de conversaciones reales
con un modelo de lenguaje, clasificadas por tarea y ocupación, producen la
primera medida de **uso efectivo** —no opinión sobre uso posible. Esta tesis
lo emplea como variable dependiente externa de su modelo de Nivel 1,
precisamente porque es el único eslabón del linaje que no fue construido por
investigadores calificando tareas.

La posición de esta tesis en ese linaje es doble. Primero, el **DBOE**
(Dynamic Benchmark-based Occupational Exposure) dota de dinámica al marco de
Felten: el vector de capacidades por aplicación deja de ser constante y se
convierte en una curva c_j(t) medida año a año sobre los resultados de la
frontera de modelos en benchmarks públicos (Epoch AI). La exposición adquiere
reloj, y su validez se establece reproduciendo el AIOE publicado con r = 0.94.
Segundo, el **DEOE** (Dynamic Embodied Occupational Exposure) construye el
espejo encarnado que el linaje no tenía: cinco subdominios físicos de O\*NET
agregados con la misma lógica de ponderación, validados de manera convergente
contra el índice de patentes robóticas de Webb (r = +0.76) y discriminante
contra el bloque cognitivo (r ≈ 0). El par DBOE–DEOE, junto con la
estructura bipolar que revela, constituye la contribución metodológica de
esta tesis: un sistema de exposición de dos fronteras, indexado en el tiempo
y anclado en capacidades medidas, aplicado a un mercado laboral en
desarrollo.

## 2.5 La adopción en el tiempo: la curva J y la ventana de política

Entre la capacidad de una tecnología y sus efectos económicos medibles media
un retraso sistemático. Brynjolfsson, Rock y Syverson (2021) lo formalizan
como la **curva J de la productividad**: las tecnologías de propósito general
exigen inversión complementaria, reorganización de procesos y aprendizaje
antes de que sus efectos aparezcan en las estadísticas, y durante ese
intervalo los indicadores pueden incluso moverse en dirección contraria a la
esperada. La historia económica es consistente: la electrificación tardó
décadas en reorganizar la fábrica; el cómputo, en aparecer en la
productividad.

Para una economía que adopta la frontera cognitiva a una fracción del ritmo
esperado —el Índice Económico de Anthropic sitúa el uso de México en 0.44
veces lo que predice su población conectada—, la teoría hace una predicción
precisa sobre el presente: **shock de capacidad sin shock laboral**. Los
efectos tempranos, cuando lleguen, aparecerán primero en los márgenes que el
modelo de tareas señala (sección 2.1): flujos de contratación, vacantes de
nivel de entrada, composición de tareas dentro del empleo —márgenes
invisibles para los acervos de empleo sectorial que miden los registros
administrativos.

De esta asimetría temporal nace el concepto organizador de la segunda mitad
de la tesis: la **ventana de política**. Si el shock de capacidad ya ocurrió,
si las expectativas de los trabajadores ya se movieron, pero el shock laboral
realizado aún no llega, el intervalo entre ambos es tiempo utilizable —finito
y medible— para reconversión, protección social y política industrial. La
ventana no es una metáfora optimista: es la lectura que la curva J impone
sobre un resultado nulo bien establecido, y el capítulo 5 la documenta con
tres piezas de evidencia independientes.

## 2.6 Mercados laborales en desarrollo: informalidad, contenido transplantado y nearshoring

Tres literaturas adaptan el marco anterior al caso mexicano.

**La informalidad como margen de ajuste.** En América Latina el trabajador
formal desplazado rara vez aparece como desempleado: reaparece como informal
(Maloney, 2004; Levy, 2008). Para esta tesis la implicación es doble. Por un
lado, metodológica: un resultado nulo medido solo sobre el empleo formal
puede enmascarar desplazamiento real; por ello la hipótesis de absorción
—¿crece la participación informal en los sectores expuestos?— se trata como
hipótesis a contrastar con microdatos de la ENOE, no como advertencia al pie.
Por otro lado, distributiva: la informalidad funciona como **multiplicador de
severidad** del mismo riesgo —idéntica exposición, cero red de protección
(sin IMSS, sin indemnización)— lo que convierte la composición formal/
informal de la población en riesgo en un dato de política de primer orden.

**El contenido de tareas varía con el desarrollo.** Trasladar descriptores
de O\*NET a México supone que una ocupación contiene tareas similares en
ambos países. Lewandowski, Park, Hardy, Du y Wu (2022) muestran que el mismo
título ocupacional es más intensivo en rutina en países de menor ingreso: el
supuesto no es inocuo. Es, sin embargo, el supuesto estándar de toda la
literatura internacional (OIT, BID, OCDE), que lo adopta sin contraste. Esta
tesis lo somete a la única prueba disponible: la comparación del perfil
ocupacional de uso observado de IA entre México y Estados Unidos, que arroja
una correlación de 0.97 —el supuesto se mantiene, se declara y se acota,
en lugar de ocultarse.

**Nearshoring y cadenas globales de valor.** La relocalización de manufactura
hacia México eleva la inversión extranjera directa —en Jalisco, +27% en
2021–2023 frente al quinquenio previo— y esa inversión llega a la frontera
global de automatización: la planta que se instala o se reequipa hoy no
replica el nivel de robotización del acervo local sino el de su corporativo
global. El nearshoring es así, simultáneamente, creación de empleo presente e
importación de adopción robótica futura: el mecanismo concreto detrás del
escenario acelerado de la curva r(t), y un factor de confusión —al alza— para
cualquier estimación del efecto temprano de la IA sobre el empleo formal
manufacturero.

## 2.7 El marco integrado

Los seis bloques anteriores se ensamblan en una cadena causal única, que es
literalmente el plan de los capítulos empíricos:

```
capacidad tecnológica (global)        §2.3, §2.4 →  c_j(t), r(t)
        ×
contenido de tareas ocupacional       §2.1       →  O*NET → DBOE, DEOE
        ×
composición local del empleo          §2.6       →  ENOE → quién está expuesto (H1, H3)
        ×
incentivo económico                   §2.2       →  IRA: factibilidad ≠ adopción (H4)
        ↓
sustitución realizada                 §2.2, §2.5 →  profundización de capital, participación
                                                    laboral, crecimiento perdido; rezagos
                                                    (ventana de política)
        ↓
incidencia distributiva               §2.6       →  educación, género, formalidad,
                                                    territorio; multiplicadores de severidad
```

Cada eslabón es medible con datos existentes, y cada uno corresponde a una
sección de los capítulos 4 a 6.

## 2.8 Hipótesis

Las hipótesis de esta tesis no son afirmaciones ad hoc: cada una es la
predicción de un bloque teórico específico.

**H1 (gradiente cognitivo).** *La exposición a los modelos de lenguaje sigue
la jerarquía ocupacional: es máxima en las ocupaciones directivas y
profesionales y mínima en las elementales y manuales.* Se deriva de §2.1 y
§2.3: la capacidad de los LLM se concentra en tareas intensivas en
habilidades cognitivas, invirtiendo el supuesto de Frey-Osborne de que el
trabajo cognitivo no rutinario constituye la zona segura.

**H2 (estructura bipolar).** *Las exposiciones cognitiva y encarnada no son
dos riesgos independientes sino polos opuestos de una dimensión dominante:
las ocupaciones más expuestas a los LLM son las menos expuestas a los robots,
y viceversa.* Se deriva de §2.3: las dos fronteras valoran las habilidades en
direcciones opuestas.

**H3 (posición de Jalisco).** *La composición del empleo de Jalisco sitúa al
estado por debajo del centro de masa de la frontera cognitiva y por encima
del de la frontera encarnada.* Se deriva de §2.3 y §2.6. El contraste
empírico nacional (capítulo 4) refina esta hipótesis: dentro de México,
Jalisco resulta además uno de los estados más próximos al polo cognitivo —una
posición **bifrontal** que enfrenta ambas fronteras a la vez.

**H4 (moderación económica).** *La exposición técnica se traduce en presión
real de automatización solo donde la relación salario/costo de capital hace
rentable la sustitución; los sectores con mayor incentivo exhiben sustitución
observable de trabajo por capital.* Se deriva de §2.2, y es la hipótesis que
distingue a esta tesis de la literatura de exposición pura: no pregunta solo
dónde *puede* automatizar el mercado, sino dónde, de hecho, *lo ha venido
haciendo*.

El capítulo siguiente describe los datos y métodos con que estas cuatro
hipótesis —y los supuestos del marco— se someten a prueba.
