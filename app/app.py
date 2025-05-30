import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    roc_curve, auc, classification_report,
    mean_squared_error, r2_score, silhouette_score
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import statsmodels.api as sm

st.set_page_config(page_title="🔬 Diagnostic Intelligent du Diabète", layout="wide")

# -------- Chargement des données --------
@st.cache_data

def load_data():
    return pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")

st.sidebar.title("⚙️ Paramètres de l'application")
uploaded_file = st.sidebar.file_uploader("📁 Importer un fichier de données (.csv)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Données chargées avec succès")
else:
    data = load_data()
    st.sidebar.info("💡 Jeu de données par défaut utilisé")

# -------- Prétraitement --------
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# -------- Navigation --------
menu = st.sidebar.radio("🧭 Menu de Navigation", [
    "🏠 Introduction au Projet",
    "📋 Exploration des Données",
    "📉 Visualisations Statistiques",
    "🧹 Nettoyage et Prétraitement",
    "🤖 Modèles de Prédiction",
    "🧬 Analyse Non Supervisée",
    "🧾 Simulation de Diagnostic"
])

# -------- Accueil --------
if menu == "🏠 Introduction au Projet":
    st.title("🩺 Système Intelligent de Détection du Diabète")
    st.markdown("""
        Cette application vous permet d'explorer un jeu de données médicales et d'appliquer des techniques d'analyse de données
        et de Machine Learning pour :

        - 🧪 Analyser des tendances dans les données cliniques
        - 🧠 Entraîner et comparer des modèles prédictifs
        - 🔍 Identifier des groupes de patients similaires
        - 🧾 Simuler un diagnostic personnalisé

        **Technologies :** Python, Streamlit, Scikit-learn, Matplotlib, Seaborn
    """)

# -------- Exploration --------
elif menu == "📋 Exploration des Données":
    st.title("📋 Exploration initiale du jeu de données")
    st.subheader("Aperçu rapide")
    st.dataframe(data.head(10))

    st.subheader("Statistiques descriptives")
    st.dataframe(data.describe())

    st.subheader("Informations structurelles")
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.subheader("Vérification des valeurs nulles ou aberrantes")
    nulls = data.isnull().sum()
    zeros = (data == 0).sum()
    st.write("**Valeurs nulles :**")
    st.dataframe(nulls)
    st.write("**Valeurs égales à zéro :**")
    st.dataframe(zeros)

# -------- Visualisation --------
elif menu == "📉 Visualisations Statistiques":
    st.title("📉 Visualisation des tendances médicales")

    st.subheader("Histogramme par variable et état diabétique")
    var = st.selectbox("Choisir une variable à explorer", X.columns)
    fig, ax = plt.subplots()
    sns.histplot(data=data, x=var, hue="Outcome", kde=True, palette="Set1", ax=ax)
    st.pyplot(fig)

    st.subheader("Carte de corrélation entre variables")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)

    st.subheader("Distribution par boxplots")
    selected = st.multiselect("Variables à afficher", X.columns.tolist(), default=X.columns[:4])
    if selected:
        fig_box, ax_box = plt.subplots(figsize=(15, 6))
        data[selected].boxplot(ax=ax_box)
        st.pyplot(fig_box)

# -------- Nettoyage --------
elif menu == "🧹 Nettoyage et Prétraitement":
    st.title("🧹 Nettoyage et prétraitement des données")

    # 📋 Imputation strategy summary
    imputation_table = pd.DataFrame({
        "Variable": ["Glucose", "BloodPressure", "BMI", "SkinThickness", "Insulin"],
        "Méthode": [
            "Moyenne par classe (Outcome)",
            "Médiane",
            "Médiane",
            "KNN",
            "KNN"
        ],
        "Justification": [
            "Fortement corrélé au risque de diabète",
            "Tendance centrale ; pas fortement dépendante de la classe",
            "Distribution asymétrique (skewed)",
            "Corrélé avec BMI/âge ; meilleure cohérence multi-variable",
            "Relations complexes et taux de valeurs manquantes élevé"
        ]
    })

    st.subheader("📌 Méthodes d’imputation par variable")
    st.table(imputation_table)
  
    # Load your dataset
    data_clean = data.copy()

    # Columns with biologically implausible zeros
    cols_to_clean = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    # Replace 0 with NaN for cleaning
    data_clean[cols_to_clean] = data_clean[cols_to_clean].replace(0, np.nan)

    # 1. Impute 'Glucose' with mean by class
    data_clean["Glucose"] = data_clean.groupby("Outcome")["Glucose"].transform(lambda x: x.fillna(x.mean()))

    # 2. Impute 'BloodPressure' and 'BMI' with median
    for col in ["BloodPressure", "BMI"]:
        data_clean[col] = data_clean[col].fillna(data_clean[col].median())

    # 3. Impute 'SkinThickness' and 'Insulin' using KNN
    knn_cols = ["SkinThickness", "Insulin"]
    knn_imputer = KNNImputer(n_neighbors=5)
    data_clean[knn_cols] = knn_imputer.fit_transform(data_clean[knn_cols])

    # Confirm no more missing values
    assert data_clean.isnull().sum().sum() == 0

    #visualisation of distributions after cleaning
    st.subheader("Distribution par boxplots après nettoyage")
    fig_box, axes = plt.subplots(2, 3, figsize=(15, 10))
    for col, ax in zip(cols_to_clean, axes.flatten()):
        sns.boxplot(data=data_clean, y=col, ax=ax)
        ax.set_title(col)
    plt.suptitle("Boxplots après nettoyage des données", fontsize=14)
    plt.tight_layout()
    st.pyplot(fig_box)

    st.subheader("📌 Séparation de données en base d'apprentissage et base de test")
    # Separate features and target
    X = data_clean.drop("Outcome", axis=1)
    y = data_clean["Outcome"]

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    
    # Choose test size percentage
    test_size_percent = st.selectbox(
        "Choisir la taille du jeu de test",
        options=[20, 30, 40],
        format_func=lambda x: f"{x}%"  # Display as %
    )

    # Convert percentage to float for train_test_split
    test_size_ratio = test_size_percent / 100

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=test_size_ratio,
        random_state=42,
        stratify=y
    )

    # Display split sizes
    st.write(f"**Nombre d'observations dans la base de test :** {len(X_test)}")
    st.write(f"**Nombre d'observations dans la base d'apprentissage :** {len(X_train)}")

# -------- Modélisation --------
elif menu == "🤖 Modèles de Prédiction":
    st.title("🤖 Évaluation des modèles de classification")

    # Initialize results dictionary to store model evaluation metrics
    if "results" not in st.session_state:
        st.session_state["results"] = {}

    # --- Model selection dropdown
    model_choice = st.selectbox(
        "Sélection du modèle",
        ["Régression Logistique", "Random Forest", "SVM", "Comparaison des Modèles"]
    )

    def store_results(name, model, acc_train, acc_test, overfit_gap, y_pred_test, proba_test, report):
        st.session_state["results"][name] = {
            "model": model,
            "accuracy_train": acc_train,
            "accuracy_test": acc_test,
            "overfit_gap": overfit_gap,
            "y_pred": y_pred_test,
            "proba": proba_test,
            "report": report
        }

    if model_choice == "Comparaison des Modèles":
        # If no models trained yet:
        if not st.session_state["results"]:
            st.warning("Aucun modèle n'a été entraîné pour comparer.")
        else:
            st.subheader("📊 Comparaison des modèles")

            # Prepare comparison dataframe
            comp_df = pd.DataFrame([
                {
                    "Modèle": k,
                    "Précision Train": v["accuracy_train"],
                    "Précision Test": v["accuracy_test"],
                    "Écart (Surapprentissage)": v["overfit_gap"]
                }
                for k, v in st.session_state["results"].items()
            ])
            st.dataframe(comp_df.style.format({
                "Précision Train": "{:.3f}",
                "Précision Test": "{:.3f}",
                "Écart (Surapprentissage)": "{:.3f}"
            }))

            # Bar chart accuracy test
            st.bar_chart(comp_df.set_index("Modèle")[["Précision Test", "Précision Train"]])

    else:
        # Show parameter widgets depending on model
        if model_choice == "Random Forest":
            st.subheader("⚙️ Paramètres du Random Forest")
            n_estimators = st.slider("Nombre d'arbres (n_estimators)", 10, 500, 100, step=10)
            max_depth = st.slider("Profondeur max des arbres (max_depth)", 1, 50, 10)
            min_samples_split = st.slider("Nombre min d'échantillons pour splitter (min_samples_split)", 2, 20, 2)

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )

        elif model_choice == "Régression Logistique":
            st.subheader("⚙️ Paramètres de la Régression Logistique")
            max_iter = st.number_input("Nombre max d'itérations (max_iter)", min_value=1000, max_value=50000, value=1000, step=100)

            model = LogisticRegression(max_iter=max_iter)

        elif model_choice == "SVM":
            st.info("Pas de paramètre interactif pour SVM pour l'instant.")
            model = SVC(probability=True)

        
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)
        proba_test = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        acc_test = accuracy_score(y_test, y_pred_test)
        acc_train = accuracy_score(y_train, y_pred_train)
        overfit_gap = acc_train - acc_test
        report = classification_report(y_test, y_pred_test, output_dict=True)

        # Store results for comparison
        store_results(model_choice, model, acc_train, acc_test, overfit_gap, y_pred_test, proba_test, report)
        st.subheader("📊 Métriques")
        st.write(f"**🎯 Précision sur test :** {acc_test:.2f}")
        st.write(f"**📘 Précision sur train :** {acc_train:.2f}")
        st.write(f"**📉 Écart (surapprentissage possible)** : {overfit_gap:.2f}")

        st.subheader("📑 Rapport de classification")
        st.dataframe(pd.DataFrame(report).transpose())

        st.subheader("🧮 Matrice de confusion")
        cm = confusion_matrix(y_test, y_pred_test)
        fig_cm, ax_cm = plt.subplots(figsize=(3, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm, cbar=False, annot_kws={"size": 10})
        ax_cm.set_xlabel("Prédits", fontsize=10)
        ax_cm.set_ylabel("Réels", fontsize=10)
        ax_cm.tick_params(labelsize=8)
        st.pyplot(fig_cm)

        if proba_test is not None:
            st.subheader("📈 Courbe ROC")
            fpr, tpr, _ = roc_curve(y_test, proba_test)
            roc_auc = auc(fpr, tpr)
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
            ax_roc.plot([0, 1], [0, 1], 'k--')
            ax_roc.set_title("Courbe ROC")
            ax_roc.set_xlabel("Taux Faux Positifs")
            ax_roc.set_ylabel("Taux Vrais Positifs")
            ax_roc.legend()
            st.pyplot(fig_roc)

        feature_names = X.columns if hasattr(X, 'columns') else [f"Var{i}" for i in range(X.shape[1])]
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            importances = None

        if importances is not None:
            st.subheader("📌 Importance des variables")
            df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
            df_imp = df_imp.sort_values(by="Importance", ascending=False)

            fig_imp, ax_imp = plt.subplots()
            sns.barplot(data=df_imp, x="Importance", y="Feature", ax=ax_imp, palette="crest")
            ax_imp.set_title("Importance des variables")
            st.pyplot(fig_imp)


# -------- Clustering --------
elif menu == "🧬 Analyse Non Supervisée":
    st.title("🧬 Clustering et réduction de dimension")

    X_all = np.concatenate([X_train, X_test])
    X_scaled = StandardScaler().fit_transform(X_all)

    # PCA transformation
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # --- COMMON FUNCTION
    def clustering_metrics(X, labels, centroids=None):
        mask = labels != -1
        if centroids is None:
            unique_labels = np.unique(labels[mask])
            centroids = np.array([X[labels == i].mean(axis=0) for i in unique_labels])
            predicted = np.array([centroids[label] if label != -1 else np.zeros(X.shape[1]) for label in labels])
        else:
            predicted = centroids[labels]
        mse = mean_squared_error(X[mask], predicted[mask])
        rmse = np.sqrt(mse)
        r2 = r2_score(X[mask], predicted[mask])
        return mse, rmse, r2

    st.header("1. Classification Ascendante Hiérarchique (CAH)")

    # --- Dendrogram construction (for visualization and cut)
    linkage_method = st.selectbox("Méthode de linkage (CAH)", ['ward', 'complete', 'average', 'single'])
    linkage_matrix = sch.linkage(X_scaled, method=linkage_method)

    st.subheader("🔍 Dendrogramme avec coupure interactive")

    # --- Distance threshold slider
    max_d = np.max(linkage_matrix[:, 2])
    distance_threshold = st.slider("Distance de coupure", 0.1, float(np.round(max_d, 2)), float(np.round(max_d / 3, 2)), 0.1)

    # --- Plot dendrogram with cutoff line
    fig_dendro, ax_dendro = plt.subplots(figsize=(10, 5))
    sch.dendrogram(linkage_matrix, ax=ax_dendro)
    ax_dendro.axhline(y=distance_threshold, color='red', linestyle='--', label=f"Seuil = {distance_threshold:.2f}")
    ax_dendro.set_title("Dendrogramme (avec seuil de coupure)")
    ax_dendro.legend()
    st.pyplot(fig_dendro)

    # --- Agglomerative Clustering with auto-cluster count based on threshold
    model_cah = AgglomerativeClustering(distance_threshold=distance_threshold, n_clusters=None, linkage=linkage_method)
    labels_cah = model_cah.fit_predict(X_scaled)
    n_found = len(set(labels_cah)) - (1 if -1 in labels_cah else 0)

    st.success(f"📌 Nombre de clusters détectés : **{n_found}**")

    if n_found >= 2:
        sil_score = silhouette_score(X_scaled, labels_cah)
        mse_cah, rmse_cah, r2_cah = clustering_metrics(X_scaled, labels_cah)
        
        st.subheader("📊 Métriques CAH")
        st.write(f"- **Silhouette Score** : {sil_score:.4f}")
        st.write(f"- **MSE** : {mse_cah:.4f}")
        st.write(f"- **RMSE** : {rmse_cah:.4f}")
        st.write(f"- **R² Score** : {r2_cah:.4f}")
    else:
        st.warning("Moins de 2 clusters détectés — les métriques ne sont pas calculables.")

    # --- PCA Visualisation
    st.subheader("📈 Visualisation des clusters (PCA)")
    fig_cah_pca, ax_cah_pca = plt.subplots()
    ax_cah_pca.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_cah, cmap='Set2')
    ax_cah_pca.set_title(f"Clusters CAH (coupé à {distance_threshold:.2f}) - PCA")
    st.pyplot(fig_cah_pca)


    # =======================
    # 🔹 2. KMEANS
    # =======================
    st.header("KMeans Clustering")

    st.subheader("Paramètres KMeans")
    k = st.number_input("Nombre de clusters (KMeans)", min_value=2, max_value=10, value=3)

    kmeans = KMeans(n_clusters=k, random_state=42)
    labels_km = kmeans.fit_predict(X_scaled)
    centroids_km = kmeans.cluster_centers_

    mse_km, rmse_km, r2_km = clustering_metrics(X_scaled, labels_km, centroids_km)
    st.subheader("📊 Métriques KMeans")
    st.write(f"- **MSE** : {mse_km:.4f}")
    st.write(f"- **RMSE** : {rmse_km:.4f}")
    st.write(f"- **R² Score** : {r2_km:.4f}")

    st.subheader("📈 Clusters KMeans - PCA")
    fig_km, ax_km = plt.subplots()
    ax_km.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_km, cmap='viridis')
    ax_km.set_title(f"Clusters KMeans (k={k}) - PCA")
    st.pyplot(fig_km)

    # =======================
    # 🔹 3. DBSCAN
    # =======================
    st.header("3. DBSCAN Clustering")

    st.subheader("Paramètres DBSCAN")
    col1, col2 = st.columns(2)
    with col1:
        eps = st.slider("Epsilon (eps)", 0.1, 5.0, 1.0, 0.1)
    with col2:
        min_samples = st.slider("Min samples", 1, 20, 8)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels_db = dbscan.fit_predict(X_scaled)

    st.subheader("📊 Métriques DBSCAN")
    if len(set(labels_db)) > 1 and any(l != -1 for l in labels_db):
        mse_db, rmse_db, r2_db = clustering_metrics(X_scaled, labels_db)
        st.write(f"- **MSE** : {mse_db:.4f}")
        st.write(f"- **RMSE** : {rmse_db:.4f}")
        st.write(f"- **R² Score** : {r2_db:.4f}")
    else:
        st.warning("DBSCAN n'a détecté qu'un seul cluster ou uniquement du bruit. Métriques non applicables.")

    st.subheader("📈 Clusters DBSCAN - PCA")
    fig_db, ax_db = plt.subplots()
    ax_db.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_db, cmap='tab10')
    ax_db.set_title(f"Clusters DBSCAN (eps={eps}, min_samples={min_samples}) - PCA")
    st.pyplot(fig_db)


# -------- Prédiction --------
elif menu == "🧾 Simulation de Diagnostic":
    st.title("🧾 Simulation de Diagnostic pour un Patient")

    model_name = st.selectbox("Choisir un modèle de prédiction", ["Régression Logistique", "Random Forest", "SVM"])
    if model_name == "Régression Logistique":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Random Forest":
        model = RandomForestClassifier()
    else:
        model = SVC(probability=True)

    model.fit(X_train, y_train)

    st.write("Veuillez entrer les informations du patient :")
    user_input = [st.slider(col, float(X[col].min()), float(X[col].max()), float(X[col].mean())) for col in X.columns]

    if st.button("Lancer la prédiction"):
        input_array = scaler.transform([user_input])
        pred = model.predict(input_array)[0]
        prob = model.predict_proba(input_array)[0][1]

        st.markdown(f"## 🩺 Diagnostic : {'🟥 Diabétique' if pred == 1 else '🟩 Non diabétique'}")
        st.markdown(f"### 🔢 Probabilité estimée : **{prob:.2%}**")
        st.progress(prob, text=f"{int(prob*100)}% de probabilité d'être diabétique")
