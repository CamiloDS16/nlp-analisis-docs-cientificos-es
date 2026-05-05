"""
Verifica integridad del consolidado antes de usarlo para entrenamiento.

Uso:
    python scripts/verificar_consolidado.py consolidado_v1_fixed.csv
    python scripts/verificar_consolidado.py consolidado_tarea_2.csv --split-col TIPO_DATASET_T2
"""

import sys
import argparse
import pandas as pd


def verificar(path: str, split_col: str, eval_val: str, train_val: str) -> bool:
    df = pd.read_csv(path, index_col=0 if path.endswith("tarea_2.csv") else False)
    print(f"\nDataset: {path}")
    print(f"Filas: {len(df):,}  |  Columna de split: {split_col}")
    print(f"Valores únicos en '{split_col}': {sorted(df[split_col].dropna().unique())}\n")

    if split_col not in df.columns:
        print(f"ERROR: columna '{split_col}' no encontrada.")
        return False

    eval_ids  = set(df[df[split_col] == eval_val]["doc_id"])
    train_ids = set(df[df[split_col].isin([train_val, "TEST"]) if "," not in train_val
                       else df[split_col].isin(train_val.split(","))]["doc_id"])

    overlap = eval_ids & train_ids
    ok = True

    # ── 1. Solapamiento cross-partition ───────────────────────────────────
    if overlap:
        print(f"FALLO — {len(overlap)} doc_ids en TRAIN y EVAL simultáneamente:")
        affected = (
            df[df["doc_id"].isin(overlap)]
            .groupby("label" if "label" in df.columns else split_col)
            .size()
            .rename("filas_afectadas")
        )
        print(affected.to_string())
        print()
        ok = False
    else:
        print(f"OK — Solapamiento TRAIN ↔ EVAL: 0 doc_ids\n")

    # ── 2. Fragmentos de texto idénticos en particiones distintas ─────────
    if "text" in df.columns:
        dup_text = (
            df.groupby("text")["doc_id"]
            .nunique()
            .pipe(lambda s: s[s > 1])
        )
        cross_text = 0
        for text, _ in dup_text.items():
            parts = set(df[df["text"] == text][split_col].unique())
            if eval_val in parts and any(v in parts for v in [train_val, "TEST", "TRAIN"]):
                cross_text += 1
        if cross_text:
            print(f"FALLO — {cross_text} fragmento(s) de texto idéntico aparecen en TRAIN y EVAL\n")
            ok = False
        else:
            print(f"OK — Texto idéntico cross-partition: 0\n")

    # ── 3. Doc_ids dentro de la misma partición (normal, solo informativo) ─
    intra = df[df.duplicated(subset=["doc_id", split_col], keep=False)]
    if len(intra):
        print(f"INFO — {intra['doc_id'].nunique()} doc_ids con múltiples fragmentos en la misma partición (esperado):")
        print(intra.groupby([split_col, "label"] if "label" in df.columns else [split_col]).size().to_string())
        print()

    # ── 4. Resumen por partición ──────────────────────────────────────────
    print("Distribución por partición:")
    col_label = "label" if "label" in df.columns else "etiqueta_auto"
    if col_label in df.columns:
        print(df.groupby([split_col, col_label]).size().unstack(fill_value=0).to_string())
    else:
        print(df[split_col].value_counts().to_string())

    print()
    print("RESULTADO:", "APROBADO" if ok else "REPROBADO — corregir antes de entrenar")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Ruta al CSV consolidado")
    parser.add_argument("--split-col", default="dataset_type",
                        help="Columna que indica la partición (default: dataset_type)")
    parser.add_argument("--eval-val", default="EVAL",
                        help="Valor de EVAL en la columna de split (default: EVAL)")
    parser.add_argument("--train-val", default="TRAIN_TEST",
                        help="Valor(es) de TRAIN separados por coma (default: TRAIN_TEST)")
    args = parser.parse_args()

    ok = verificar(args.path, args.split_col, args.eval_val, args.train_val)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
