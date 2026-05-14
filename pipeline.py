"""
Tech Challenge - Fase 4: Pipeline de Machine Learning para Predição de Obesidade
Height e Weight removidos para evitar data leakage (IMC vaza o target diretamente).
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
)
from sklearn.feature_selection import SelectFromModel

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "Obesity.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── 1. CARREGAMENTO ────────────────────────────────────────────────────────────
print("=" * 60)
print("1. CARREGAMENTO DOS DADOS")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"Shape original: {df.shape}")
print(f"Colunas: {df.columns.tolist()}")
print(f"\nDistribuição do target:\n{df['Obesity'].value_counts()}")

# ── 2. PRÉ-PROCESSAMENTO ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. PRÉ-PROCESSAMENTO")
print("=" * 60)

# Remover Height e Weight — data leakage (IMC → target diretamente)
df = df.drop(columns=["Height", "Weight"])
print("Height e Weight removidos (data leakage via IMC).")

# Arredondar colunas ordinais que podem ter decimais
ordinal_round = ["FCVC", "NCP", "CH2O", "FAF", "TUE"]
for col in ordinal_round:
    df[col] = df[col].round().astype(int)

print(f"Shape após remoção: {df.shape}")
print(f"Valores nulos:\n{df.isnull().sum()}")

# ── 3. FEATURE ENGINEERING ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. FEATURE ENGINEERING")
print("=" * 60)

# Mapa ordinal para CAEC e CALC (frequência de consumo)
freq_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
df["CAEC_ord"] = df["CAEC"].map(freq_map)
df["CALC_ord"] = df["CALC"].map(freq_map)

# Score composto de hábitos saudáveis
df["healthy_score"] = (
    df["FCVC"]          # mais vegetais = melhor
    + df["FAF"]         # mais exercício = melhor
    + df["CH2O"]        # mais água = melhor
    - df["CAEC_ord"]    # menos lanche = melhor
    - df["CALC_ord"]    # menos álcool = melhor
    - df["TUE"]         # menos tela = melhor
)

# Score de risco comportamental
df["risk_score"] = (
    df["CAEC_ord"]
    + df["CALC_ord"]
    + df["TUE"]
    + df["FAVC"].map({"yes": 1, "no": 0})
    + df["SMOKE"].map({"yes": 1, "no": 0})
)

print("Features criadas: healthy_score, risk_score")
print(f"Shape final: {df.shape}")

# ── 4. SEPARAÇÃO FEATURES / TARGET ────────────────────────────────────────────
TARGET = "Obesity"
DROP_COLS = ["CAEC", "CALC"]  # já codificadas em CAEC_ord / CALC_ord

X = df.drop(columns=[TARGET] + DROP_COLS)
y = df[TARGET]

# Codificar target como inteiro
le_target = LabelEncoder()
y_enc = le_target.fit_transform(y)
print(f"\nClasses: {le_target.classes_}")

# ── 5. DEFINIÇÃO DAS COLUNAS POR TIPO ─────────────────────────────────────────
binary_cols = ["Gender", "family_history", "FAVC", "SMOKE", "SCC"]
ordinal_cols = ["FCVC", "NCP", "CH2O", "FAF", "TUE", "CAEC_ord", "CALC_ord"]
nominal_cols = ["MTRANS"]
numeric_cols = ["Age", "healthy_score", "risk_score"]

# Codificação binária simples
binary_map = {"Female": 0, "Male": 1, "yes": 1, "no": 0}
for col in binary_cols:
    X[col] = X[col].map(binary_map)

# One-hot para MTRANS
X = pd.get_dummies(X, columns=nominal_cols, drop_first=True)

print(f"\nFeatures finais ({X.shape[1]}): {X.columns.tolist()}")

# ── 6. SPLIT TREINO / TESTE ───────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
print(f"\nTreino: {X_train.shape} | Teste: {X_test.shape}")

# ── 7. PIPELINES DOS MODELOS ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. TREINAMENTO DOS MODELOS")
print("=" * 60)

scaler = StandardScaler()

models = {
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ]),
    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=200, random_state=42)),
    ]),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, pipe in models.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    results[name] = scores
    print(f"{name}: CV Accuracy = {scores.mean():.4f} ± {scores.std():.4f}")

# ── 8. SELEÇÃO DO MELHOR MODELO ───────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k].mean())
best_pipe = models[best_name]
print(f"\nMelhor modelo: {best_name}")

best_pipe.fit(X_train, y_train)
y_pred = best_pipe.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Acurácia no teste: {acc:.4f} ({acc*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

# ── 9. VISUALIZAÇÕES ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. GERANDO VISUALIZAÇÕES")
print("=" * 60)

# 9.1 Comparação de modelos
fig, ax = plt.subplots(figsize=(9, 5))
names = list(results.keys())
means = [results[k].mean() for k in names]
stds = [results[k].std() for k in names]
bars = ax.barh(names, means, xerr=stds, color=["#2196F3", "#4CAF50", "#FF9800"], capsize=5)
ax.axvline(0.75, color="red", linestyle="--", label="Meta 75%")
ax.set_xlabel("Acurácia (CV)")
ax.set_title("Comparação de Modelos — Cross-Validation")
ax.legend()
for bar, mean in zip(bars, means):
    ax.text(mean + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{mean:.3f}", va="center", fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "comparacao_modelos.png", dpi=150)
plt.close()
print("Salvo: outputs/comparacao_modelos.png")

# 9.2 Matriz de confusão
fig, ax = plt.subplots(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=le_target.classes_)
disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
ax.set_title(f"Matriz de Confusão — {best_name}")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "matriz_confusao.png", dpi=150)
plt.close()
print("Salvo: outputs/matriz_confusao.png")

# 9.3 Importância das features (se Random Forest ou Gradient Boosting)
clf = best_pipe.named_steps["clf"]
if hasattr(clf, "feature_importances_"):
    feat_imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    feat_imp.plot(kind="barh", ax=ax, color="#2196F3")
    ax.set_title(f"Importância das Features — {best_name}")
    ax.set_xlabel("Importância")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "feature_importance.png", dpi=150)
    plt.close()
    print("Salvo: outputs/feature_importance.png")

# 9.4 Distribuição do target
fig, ax = plt.subplots(figsize=(10, 5))
order = le_target.classes_
df[TARGET].value_counts().reindex(order).plot(kind="bar", ax=ax, color="#4CAF50", edgecolor="black")
ax.set_title("Distribuição das Classes de Obesidade")
ax.set_xlabel("Classe")
ax.set_ylabel("Contagem")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "distribuicao_classes.png", dpi=150)
plt.close()
print("Salvo: outputs/distribuicao_classes.png")

# ── 10. EXPORTAR MODELO ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. EXPORTANDO MODELO")
print("=" * 60)

model_path = OUTPUTS_DIR / "modelo_obesidade.pkl"
joblib.dump({"model": best_pipe, "label_encoder": le_target, "feature_cols": list(X.columns)}, model_path)
print(f"Modelo salvo em: {model_path}")

# ── 11. RESUMO FINAL ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESUMO FINAL")
print("=" * 60)
print(f"Melhor modelo    : {best_name}")
print(f"Acurácia (teste) : {acc*100:.2f}%")
print(f"Meta atingida    : {'SIM' if acc >= 0.75 else 'NAO'} (meta: 75%)")
print(f"Features usadas  : {X.shape[1]} (Height e Weight excluídos)")
print(f"Saídas geradas   : {OUTPUTS_DIR}")
