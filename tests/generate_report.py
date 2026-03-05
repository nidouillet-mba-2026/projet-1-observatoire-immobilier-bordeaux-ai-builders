#!/usr/bin/env python3
"""Generateur de rapport de notation - Projet 1 : Observatoire Immobilier."""

import os
import xml.etree.ElementTree as ET

SCORE_MAP = {
    # Stats from scratch (15 pts)
    "test_mean_correct":          (3,  "Stats from scratch"),
    "test_variance_correct":      (4,  "Stats from scratch"),
    "test_correlation_correct":   (4,  "Stats from scratch"),
    "test_no_numpy_in_stats":     (4,  "Stats from scratch"),
    # Regression from scratch (10 pts)
    "test_least_squares_fit":     (5,  "Regression from scratch"),
    "test_r_squared_perfect_fit": (3,  "Regression from scratch"),
    "test_no_sklearn_in_regression": (2, "Regression from scratch"),
    # Donnees (15 pts)
    "test_dvf_exists":            (5,  "Donnees reelles"),
    "test_dvf_min_rows":          (5,  "Donnees reelles"),
    "test_annonces_exists":       (5,  "Donnees reelles"),
    # Application (10 pts)
    "test_app_file_exists":       (4,  "Application deployee"),
    "test_readme_has_deployed_url": (6, "Application deployee"),
    # Tests etudiants (5 pts)
    "test_student_tests_exist":   (5,  "Tests etudiants"),
}

SOUTENANCE_ITEMS = [
    ("Architecture et choix techniques", 15),
    ("Comprehension des algos from scratch", 10),
    ("Repartition du travail", 10),
    ("Recul critique", 5),
]


def parse_results(xml_path):
    results = {}
    if not os.path.exists(xml_path):
        return results
    root = ET.parse(xml_path).getroot()
    for tc in root.iter("testcase"):
        name = tc.get("name", "")
        failed = tc.find("failure") is not None or tc.find("error") is not None
        skipped = tc.find("skipped") is not None
        results[name] = "skip" if skipped else ("fail" if failed else "pass")
    return results


def generate_report(results):
    lines = [
        "# Rapport d'Evaluation - Projet 1 : Observatoire Immobilier",
        "",
        "## Score automatique (GitHub CI)",
        "",
    ]

    categories = list(dict.fromkeys(v[1] for v in SCORE_MAP.values()))
    total_earned = total_possible = 0

    for cat in categories:
        lines += [f"### {cat}", "", "| Critere | Points | Resultat |",
                  "|---------|--------|----------|"]
        for test, (pts, c) in SCORE_MAP.items():
            if c != cat:
                continue
            status = results.get(test, "not_run")
            icon = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}.get(status, "N/A")
            earned = pts if status == "pass" else 0
            total_earned += earned
            total_possible += pts
            display = test.replace("test_", "").replace("_", " ").title()
            lines.append(f"| {display} | {earned}/{pts} | {icon} |")
        lines.append("")

    lines += [
        "---",
        "",
        "## Soutenance (evalue par l'enseignant)",
        "",
        "| Critere | Points |",
        "|---------|--------|",
    ]
    soutenance_total = 0
    for label, pts in SOUTENANCE_ITEMS:
        lines.append(f"| {label} | ?/{pts} |")
        soutenance_total += pts
    lines += [
        "",
        "---",
        "",
        "## Resume",
        "",
        f"| | Score |",
        f"|--|-------|",
        f"| **CI automatique** | **{total_earned} / {total_possible}** |",
        f"| Soutenance (enseignant) | ? / {soutenance_total} |",
        f"| **Total** | **/ {total_possible + soutenance_total}** |",
        "",
    ]

    if total_earned == total_possible:
        lines.append("> Tous les tests CI passent.")
    elif total_earned >= total_possible * 0.7:
        lines.append("> Bon travail. Quelques tests en echec a corriger.")
    else:
        lines.append("> Plusieurs tests echouent. Consultez les details ci-dessus.")

    lines += ["", "---", "*Rapport genere automatiquement par le CI.*"]
    return "\n".join(lines)


def main():
    results = parse_results("eval_results.xml")
    report = generate_report(results)
    with open("evaluation_report.md", "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
