"""
Compile the thesis draft into a reviewable DOCX and PDF.

Assembles the chapter markdown files in reading order, inserts each figure
at the end of its mapped section (with caption), prepends a title page, and
builds:
  thesis/tesis_borrador.docx  (pandoc; with table of contents -- for the
                               advisor to comment with track changes)
  thesis/tesis_borrador.pdf   (Word COM automation from the docx)

Rerun after editing any chapter. Requires pandoc and MS Word installed.
"""

import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGDIR = (HERE.parent / "figures").as_posix()

CHAPTERS = [
    "cap1_introduccion.md",
    "cap2_marco_teorico.md",
    "cap3_datos_metodos.md",
    "cap4_resultados_estructura.md",
    "cap5_resultados_mercado.md",
    "cap6_resultados_proyeccion.md",
    "cap7_discusion.md",
    "cap8_conclusiones.md",
]

# figure -> (chapter file, section heading prefix it belongs to, caption)
FIGURES = {
    "cap4_resultados_estructura.md": [
        ("## 4.3", "worker_exposure_profile.png",
         "Figura 4.1. Exposición media ponderada por perfil del trabajador "
         "(z; ENOE 2024-T3)."),
        ("## 4.4", "state_exposure_map.png",
         "Figura 4.2. Los 32 estados sobre los dos ejes de automatización "
         "(ENOE 2024-T3; burbuja = ocupados; Jalisco resaltado)."),
        ("## 4.5", "municipal_exposure_map.png",
         "Figura 4.3. Exposición municipal: DBOE, DEOE y presión 2030 "
         "(125 municipios; mezcla sectorial formal CE 2023)."),
    ],
    "cap5_resultados_mercado.md": [
        ("## 5.1", "h4_adoption_test.png",
         "Figura 5.1. Conducta de sustitución revelada: profundización de "
         "capital vs. incentivo rezagado, y empleo 2003–2023 vs. exposición "
         "(panel censal de Jalisco)."),
        ("## 5.3", "chatgpt_event_imss.png",
         "Figura 5.2. Empleo formal (IMSS) alrededor del choque de IA "
         "generativa, por grupo de exposición cognitiva."),
        ("## 5.4", "absorption_informality.png",
         "Figura 5.3. Empleo total e informalidad por grupo de exposición "
         "alrededor del choque (ENOE, 2022-T1 a 2024-T3)."),
        ("## 5.5", "nearshoring_fdi.png",
         "Figura 5.4. IED de Jalisco por año y componente greenfield "
         "(Secretaría de Economía, 2006–2023)."),
        ("## 5.6", "perception_trend.png",
         "Figura 5.5. Expectativa de desplazamiento por robots/IA en México "
         "(Latinobarómetro, escalas armonizadas)."),
    ],
    "cap6_resultados_proyeccion.md": [
        ("## 6.3", "workers_at_risk.png",
         "Figura 6.1. Trabajadores por encima de la barra de presión alta de "
         "hoy hacia 2030, por escenario y polo dominante."),
    ],
}

TITLE = """---
lang: es
title: "Riesgo de automatización laboral por inteligencia artificial en Jalisco"
subtitle: "Exposición cognitiva y encarnada, incentivo económico y proyección 2025–2030"
author: "Carlos Pulido Rosas — Maestría en Ciencias de los Datos, CUCEA, Universidad de Guadalajara"
date: "BORRADOR PARA REVISIÓN — junio de 2026"
---

*Repositorio: github.com/Carpuro/ai-automation-risk-jalisco — todos los
resultados son reproducibles; cada cifra proviene de una tabla generada por
un script versionado.*

\\newpage

"""


def insert_figures(name: str, text: str) -> str:
    for section, fig, caption in FIGURES.get(name, []):
        lines = text.split("\n")
        start = next(i for i, l in enumerate(lines) if l.startswith(section))
        end = next((i for i in range(start + 1, len(lines))
                    if lines[i].startswith("## ")), len(lines))
        block = ["", f"![{caption}]({FIGDIR}/{fig}){{width=15.5cm}}", ""]
        lines[end:end] = block
        text = "\n".join(lines)
    return text


def main() -> None:
    parts = [TITLE]
    for name in CHAPTERS:
        text = (HERE / name).read_text(encoding="utf-8")
        parts.append(insert_figures(name, text))
        parts.append("\n\\newpage\n")
    combined = HERE / "_tesis_combined.md"
    combined.write_text("\n".join(parts), encoding="utf-8")

    docx = HERE / "tesis_borrador.docx"
    subprocess.run(
        ["pandoc", str(combined), "-o", str(docx),
         "--toc", "--toc-depth=2", "--standalone"],
        check=True)
    print(f"-> {docx}")

    # DOCX -> PDF via Word COM
    import win32com.client
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    try:
        doc = word.Documents.Open(str(docx))
        doc.Fields.Update()                       # populate the TOC
        pdf = HERE / "tesis_borrador.pdf"
        doc.SaveAs2(str(pdf), FileFormat=17)      # 17 = wdFormatPDF
        doc.Close(False)
        print(f"-> {pdf}")
    finally:
        word.Quit()


if __name__ == "__main__":
    main()
