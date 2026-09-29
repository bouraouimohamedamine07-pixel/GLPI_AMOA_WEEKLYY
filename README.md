# BKFI AMOA — GLPI Weekly Dashboard

Application Streamlit prête à lancer dans VS Code.

## 1. Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Lancement

```bash
streamlit run app.py
```

Le navigateur s'ouvre sur le dashboard.

## 3. Première utilisation

Dans la barre latérale :

1. Charger le nouvel export GLPI `.xlsx`.
2. Optionnellement charger `BKFI_AMOA_suivi GLPI.xlsx` pour récupérer les champs métier existants de `Action_Log`.
3. Cliquer **Importer / Actualiser**.

La base locale `glpi_dashboard.db` est créée automatiquement.

## 4. Principe important

L'ID GLPI est la clé du ticket. Lors d'un nouvel import, les champs provenant de GLPI sont actualisés mais les champs métier (`Action`, `Blocage`, `Jira`, `Responsable`, `Deadline`, `Statut opérationnel`, `Commentaire`, etc.) sont conservés.

## 5. Classification automatique

Le référentiel métier reprend :

- Anomalie
- Amélioration
- Nouvelle demande
- Réglementaire / conformité
- Technique
- Data & reporting
- Formation
- Correction
- Accès

Les mots-clés sont modifiables directement depuis **Référentiel / IA**. La classification analyse le titre + la description du ticket et donne une confiance. Une correction manuelle peut être enregistrée via `Function` dans l'Action Log.

## 6. Analyse type TCD

La page **Analyse TCD** permet de choisir les champs en lignes et colonnes et de produire une vue pivot sans modifier le code.

## 7. Évolution prévue

Cette première version constitue le moteur fonctionnel. La prochaine étape peut remplacer Streamlit par React + FastAPI tout en gardant la même base et les mêmes règles métier.
