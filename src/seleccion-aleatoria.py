import random
import csv

TOTAL_DOCS = 1_342_100
GRUPO = ["Jesus", "Camilo", "Mateo", "Sergio"]
SEED = 42  

# indices para los docs txt
indices = list(range(1, TOTAL_DOCS + 1))


random.seed(SEED)
random.shuffle(indices)


chunk = TOTAL_DOCS // len(GRUPO)

particiones = {
    GRUPO[0]: indices[0          : chunk],
    GRUPO[1]: indices[chunk      : chunk*2],
    GRUPO[2]: indices[chunk*2    : chunk*3],
    GRUPO[3]: indices[chunk*3    : ],  
}

# se crea archivo csv con la division aleatoria
with open("particiones.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["doc_id", "asignado_a"])
    for nombre, ids in particiones.items():
        for doc_id in ids:
            writer.writerow([f"{doc_id:07d}", nombre])

# verificacion de no solapamiento
todos = [id for ids in particiones.values() for id in ids]
assert len(todos) == len(set(todos)) == TOTAL_DOCS
print("\nVerificacion OK: no hay solapamiento")