import pandas as pd
import matplotlib.pyplot as plt

# Carga los resultados
df = pd.read_csv("runs/detect/train5/results.csv")

# Grafica box_loss entrenamiento vs validación
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['metrics/mAP50(B)'], label='MAP50', color='blue')
#plt.plot(df.index, df['val/box_loss'], label='Box Loss (val)', color='orange')

plt.xlabel("Epoch")
plt.ylabel("Box Loss")
plt.title("Box Loss: Entrenamiento vs Validación")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("map50.png")