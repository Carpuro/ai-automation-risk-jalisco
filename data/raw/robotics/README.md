# Robotics adoption data — sources

Inputs for the DEOE dynamic layer (`build_robot_capability.py`). All values
come from primary IFR World Robotics publications (free executive summaries /
press releases, PDFs archived in this folder) or from Our World in Data's
redistribution of the IFR series.

## World series (OWID, source IFR)

- `owid_robots_global_stock.csv` — world annual installations + operational
  stock 2012–2024. Downloaded 2026-06-10 from
  https://ourworldindata.org/grapher/industrial-robots-annual-installations-total-operational
- `owid_robots_installed.csv`, `owid_robot_density.csv` — top-country series
  (no Mexico; kept for reference).

## Mexico annual installations (IFR executive summaries, archived here)

| Year | Units | Source (data year = report year − 1) |
|---|---|---|
| 2019 | 4,600 | IFR via The Robot Report; consistent with WR2021's −26% for 2020 |
| 2020 | 3,363 | `IFR_Executive_Summary_WR_2021.pdf`, p. "Americas" (−26%) |
| 2021 | 5,401 | `IFR_Executive_Summary_WR_2022.pdf` (+61%) |
| 2022 | 6,000 | `IFR_Executive_Summary_WR_Industrial_Robots_2023.pdf` (+13%) |
| 2023 | 5,832 | `IFR_Executive_Summary_WR_2024.pdf` (−3%) |
| 2024 | 5,594 | `IFR_Executive_Summary_WR_2025.pdf` (−4%) |

Notes: WR2025 prose says 5,594 units; the IFR press release for the same
edition rounds to 5,600. Automotive accounts for 63–70% of Mexican
installations across these years. Pre-2019 values exist only as a chart in
`IFR_Outlook_2019_Chicago.pdf` (image, not extractable); IFR notes 2018
"dropped back to 2015 level".

Country-level operational stock and robot density for Mexico are only in the
paid IFR report — not used.
