"""
calcular_kappa.py
=================
Calcula Cohen's Kappa y Krippendorff's Alpha entre dos anotadores
a partir de los exports JSON de Label Studio.

Uso:
  python scripts/calcular_kappa.py \
      --anotador1 label_studio/export_camilo.json \
      --anotador2 label_studio/export_companero.json

Cómo exportar desde Label Studio:
  Export > JSON > descargar y guardar en label_studio/
"""

import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def cargar_anotaciones(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    anotaciones = {}
    for tarea in data:
        fid = tarea["data"]["fragmento_id"]
        completions = tarea.get("annotations", [])
        if not completions:
            continue
        # Tomar la primera anotación de cada tarea
        resultado = completions[0].get("result", [])
        etiqueta = None
        for r in resultado:
            if r.get("from_name") == "etiqueta":
                etiqueta = r["value"]["choices"][0]
                break
        if etiqueta:
            anotaciones[fid] = etiqueta

    return anotaciones


def krippendorff_alpha(a1: list, a2: list, labels: list) -> float:
    n = len(a1)
    label_idx = {l: i for i, l in enumerate(labels)}
    v1 = [label_idx[x] for x in a1]
    v2 = [label_idx[x] for x in a2]

    # Observed disagreement
    do = sum(1 for x, y in zip(v1, v2) if x != y) / n

    # Expected disagreement
    all_vals = v1 + v2
    counts = np.bincount(all_vals, minlength=len(labels))
    probs = counts / counts.sum()
    de = 1 - sum(p ** 2 for p in probs)

    if de == 0:
        return 1.0
    return 1 - do / de


def main(path1: str, path2: str):
    ann1 = cargar_anotaciones(path1)
    ann2 = cargar_anotaciones(path2)

    comunes = sorted(set(ann1.keys()) & set(ann2.keys()))
    if not comunes:
        print("ERROR: no hay fragmento_ids en común entre los dos exports.")
        return

    print(f"Fragmentos anotados por ambos: {len(comunes)}")
    print(f"  Solo en anotador 1: {len(ann1) - len(comunes)}")
    print(f"  Solo en anotador 2: {len(ann2) - len(comunes)}")

    y1 = [ann1[fid] for fid in comunes]
    y2 = [ann2[fid] for fid in comunes]

    labels = sorted(set(y1 + y2))

    kappa = cohen_kappa_score(y1, y2)
    alpha = krippendorff_alpha(y1, y2, labels)

    print(f"\nCohen's Kappa:         {kappa:.4f}")
    print(f"Krippendorff's Alpha:  {alpha:.4f}")

    if kappa >= 0.8:
        nivel = "Casi perfecto"
    elif kappa >= 0.6:
        nivel = "Sustancial"
    elif kappa >= 0.4:
        nivel = "Moderado"
    else:
        nivel = "Pobre — revisar guía de anotación"
    print(f"Nivel de acuerdo:      {nivel}")

    # Por etiqueta
    df = pd.DataFrame({"fragmento_id": comunes, "ann1": y1, "ann2": y2})
    print("\nAcuerdo por etiqueta:")
    for etiqueta in labels:
        sub = df[df["ann1"] == etiqueta]
        if sub.empty:
            continue
        acuerdo = (sub["ann1"] == sub["ann2"]).mean()
        print(f"  {etiqueta:6s}: {acuerdo:.1%} acuerdo ({len(sub)} fragmentos)")

    # Desacuerdos
    desacuerdos = df[df["ann1"] != df["ann2"]]
    if not desacuerdos.empty:
        print(f"\nDesacuerdos ({len(desacuerdos)}):")
        print(desacuerdos[["fragmento_id", "ann1", "ann2"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--anotador1", required=True)
    parser.add_argument("--anotador2", required=True)
    args = parser.parse_args()
    main(args.anotador1, args.anotador2)
