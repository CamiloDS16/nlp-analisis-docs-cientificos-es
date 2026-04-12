"""
prepare_label_studio.py
=======================
Toma camilo_dataset_A.parquet (descargado de Drive) y produce:

  data/processed/task1/dataset_A_raw.parquet     — copia del original (no se modifica)
  data/processed/task1/validacion_LIM.parquet    — 200 LIM para anotar manualmente
  data/processed/task1/validacion_CONC.parquet   — 200 CONC para anotar manualmente
  data/processed/task1/train_LIM.parquet         — LIM restantes para entrenamiento
  data/processed/task1/train_CONC.parquet        — CONC restantes para entrenamiento
  label_studio/tasks_validacion.json             — archivo de importación para Label Studio

Uso:
  python scripts/prepare_label_studio.py --input /ruta/a/camilo_dataset_A.parquet
"""

import argparse
import json
import os
import pandas as pd

SEED = 42
N_VAL = 200  # fragmentos por etiqueta para validación humana

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK1_DIR = os.path.join(BASE, "data", "processed", "task1")
LS_DIR = os.path.join(BASE, "label_studio")


def main(input_path: str):
    print(f"Cargando: {input_path}")
    df = pd.read_parquet(input_path)

    print("\nDistribución en el archivo recibido:")
    print(df["etiqueta_auto"].value_counts().to_string())

    # Guardar copia del original sin tocar
    raw_path = os.path.join(TASK1_DIR, "dataset_A_raw.parquet")
    df.to_parquet(raw_path, index=False)
    print(f"\nCopia guardada en: {raw_path}")

    resultados = {}

    for etiqueta in ["LIM", "CONC"]:
        df_e = df[df["etiqueta_auto"] == etiqueta].copy()
        disponibles = len(df_e)
        print(f"\n{etiqueta}: {disponibles} fragmentos disponibles")

        if disponibles < N_VAL:
            print(f"  ADVERTENCIA: solo hay {disponibles} fragmentos de {etiqueta}, "
                  f"se usarán todos para validación.")
            n_val = disponibles
        else:
            n_val = N_VAL

        val = df_e.sample(n=n_val, random_state=SEED)
        train = df_e.drop(val.index)

        val_path = os.path.join(TASK1_DIR, f"validacion_{etiqueta}.parquet")
        train_path = os.path.join(TASK1_DIR, f"train_{etiqueta}.parquet")

        val.to_parquet(val_path, index=False)
        train.to_parquet(train_path, index=False)

        print(f"  Validación: {len(val)} → {val_path}")
        print(f"  Entrenamiento: {len(train)} → {train_path}")

        resultados[etiqueta] = val

    # Combinar LIM + CONC para Label Studio y mezclar
    df_val = pd.concat(resultados.values(), ignore_index=True)
    df_val = df_val.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Formato de importación de Label Studio
    tasks = []
    for _, row in df_val.iterrows():
        tasks.append({
            "data": {
                "text": row["texto"],
                "fragmento_id": str(row["fragmento_id"]),
                "doc_id": str(row["doc_id"]),
                "etiqueta_auto": row["etiqueta_auto"],
                "num_palabras": int(row["num_palabras"]) if "num_palabras" in row else None,
            }
        })

    ls_path = os.path.join(LS_DIR, "tasks_validacion.json")
    with open(ls_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"\nLabel Studio — {len(tasks)} tareas exportadas a: {ls_path}")
    print("\nResumen final:")
    print(f"  Validación total: {len(df_val)} ({len(resultados.get('LIM', []))} LIM + "
          f"{len(resultados.get('CONC', []))} CONC)")
    print(f"  Listos para importar en Label Studio.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Ruta al parquet descargado de Drive")
    args = parser.parse_args()
    main(args.input)
