import os
import re
import json
import html
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

import numpy as np

# Embeddings locaux (comprend le sens) : pip install sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    EMBED_AVAILABLE = True
except Exception:
    EMBED_AVAILABLE = False

EMBED_MODELS = {
    "Rapide (MiniLM, ~120 MB)": "paraphrase-multilingual-MiniLM-L12-v2",
    "Précis (mpnet, ~1.1 GB) — recommandé": "paraphrase-multilingual-mpnet-base-v2",
}

DEFAULT_DESC = {
    "Anomalie": "Un problème ou un bug : l'application ne fonctionne pas, plante, affiche une erreur ou un résultat incorrect",
    "Amélioration": "Demande d'évolution ou d'amélioration d'une fonctionnalité existante",
    "Nouvelle demande": "Nouveau besoin ou nouvelle fonctionnalité à créer, qui n'existe pas encore",
    "Réglementaire / conformité": "Exigence réglementaire, audit, conformité, norme ou contrôle",
    "Technique": "Problème d'infrastructure : serveur, réseau, connexion, base de données, performance, installation",
    "Data & reporting": "Demande de rapport, extraction de données, statistiques, tableau de bord, indicateurs",
    "Formation": "Besoin de formation, d'explication, de documentation ou d'accompagnement pour utiliser l'outil",
    "Correction": "Correction de données erronées ou d'une erreur de saisie dans le système",
    "Accès": "Demande d'accès, habilitation, droits, compte, mot de passe ou permissions",
}

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "glpi_dashboard.db"
DEFAULT_MASTER = APP_DIR / "BKFI_AMOA_suivi GLPI.xlsx"

RAW_PREFIX = "Brut · "

st.set_page_config(
    page_title="BKFI AMOA — GLPI Weekly",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Database
# -----------------------------

def db():
    return sqlite3.connect(DB_PATH)


def init_db():
    with db() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS tickets (
            id TEXT PRIMARY KEY,
            titre TEXT,
            entite TEXT,
            statut_glpi TEXT,
            description TEXT,
            date_ouverture TEXT,
            priorite TEXT,
            demandeur TEXT,
            technicien TEXT,
            categorie_glpi TEXT,
            derniere_modif TEXT,
            function_auto TEXT,
            function_override TEXT,
            action TEXT,
            blocage TEXT,
            jira TEXT,
            responsable TEXT,
            deadline TEXT,
            statut_operationnel TEXT,
            date_cloture TEXT,
            commentaire TEXT,
            first_seen TEXT,
            last_import TEXT,
            raw_json TEXT
        )""")

        # Migration : ajoute raw_json si l'ancienne table ne l'a pas
        tcols = [r[1] for r in con.execute("PRAGMA table_info(tickets)")]
        if "raw_json" not in tcols:
            con.execute("ALTER TABLE tickets ADD COLUMN raw_json TEXT")

        con.execute("""CREATE TABLE IF NOT EXISTS snapshots (
            import_id INTEGER PRIMARY KEY AUTOINCREMENT,
            imported_at TEXT,
            source_file TEXT,
            ticket_count INTEGER
        )""")

        con.execute("""CREATE TABLE IF NOT EXISTS snapshot_rows (
            import_id INTEGER,
            ticket_id TEXT,
            statut_glpi TEXT,
            categorie_glpi TEXT,
            function_auto TEXT,
            responsable TEXT,
            technicien TEXT
        )""")

        # Migration : ajoute la colonne technicien si l'ancienne table ne l'a pas
        cols = [r[1] for r in con.execute("PRAGMA table_info(snapshot_rows)")]
        if "technicien" not in cols:
            con.execute("ALTER TABLE snapshot_rows ADD COLUMN technicien TEXT")

        con.execute("""CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )""")

        con.execute("""CREATE TABLE IF NOT EXISTS rules (
            label TEXT PRIMARY KEY,
            keywords TEXT
        )""")


def get_setting(key, default=""):
    with db() as con:
        r = con.execute(
            "SELECT value FROM settings WHERE key=?", (key,)
        ).fetchone()
    return r[0] if r else default


def set_setting(key, value):
    with db() as con:
        con.execute(
            "INSERT OR REPLACE INTO settings(key, value) VALUES (?,?)",
            (key, value)
        )


def get_desc(label):
    return get_setting("desc:" + label, DEFAULT_DESC.get(label, ""))


def get_embed_model_name():
    cur = get_setting("embed_model", list(EMBED_MODELS)[0])
    return cur if cur in EMBED_MODELS else list(EMBED_MODELS)[0]


DEFAULT_RULES = {
    "Anomalie": [
        "bug", "erreur", "anomalie", "dysfonctionnement",
        "bloqué", "blocage", "ne fonctionne", "message d'erreur"
    ],
    "Amélioration": [
        "amélioration", "optimiser", "optimisation",
        "évolution", "ajout", "améliorer", "amélioré"
    ],
    "Nouvelle demande": [
        "nouvelle demande", "nouveau besoin",
        "demande de création", "besoin de",
        "souhaite mettre en place"
    ],
    "Réglementaire / conformité": [
        "réglementaire", "conformité", "audit",
        "obligation", "loi", "norme", "contrôle réglementaire"
    ],
    "Technique": [
        "serveur", "réseau", "connexion", "base de données",
        "sql", "performance", "infrastructure",
        "système", "installation"
    ],
    "Data & reporting": [
        "reporting", "rapport", "report", "état",
        "statistique", "dashboard", "tableau de bord",
        "requête", "donnée", "données", "excel",
        "power bi", "indicateur", "kpi"
    ],
    "Formation": [
        "formation", "former", "apprentissage",
        "documentation", "guide utilisateur",
        "manuel", "explication", "accompagnement",
        "comment utiliser"
    ],
    "Correction": [
        "corriger", "correction", "rectifier",
        "mauvaise donnée", "erreur de saisie",
        "donnée incorrecte", "modifier une erreur"
    ],
    "Accès": [
        "accès", "acces", "habilitation", "habilitations",
        "droit", "droits", "permission", "permissions",
        "compte", "mot de passe", "profil", "autorisation"
    ],
}


def load_rules():
    with db() as con:
        rows = con.execute(
            "SELECT label, keywords FROM rules"
        ).fetchall()

    if rows:
        return {
            r[0]: [
                x.strip().lower()
                for x in r[1].split(",")
                if x.strip()
            ]
            for r in rows
        }

    return {k: list(v) for k, v in DEFAULT_RULES.items()}


def save_rules(rules):
    with db() as con:
        for label, words in rules.items():
            con.execute(
                "INSERT OR REPLACE INTO rules(label, keywords) VALUES (?,?)",
                (label, ",".join(words))
            )


def norm(v):
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v).strip()


def clean_text(v):
    """Nettoie le HTML GLPI (&lt;p&gt;, <br>, &nbsp;...) → texte brut."""
    s = norm(v)
    if not s:
        return ""
    s = html.unescape(s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# -----------------------------
# Classification
# -----------------------------

def classify(text, rules):
    """Étape 1 : classification par mots-clés (sur titre + description nettoyée)."""
    text = clean_text(text).lower()

    if not text:
        return "À vérifier", 0.0

    scores = {}

    for label, words in rules.items():
        score = 0

        for w in words:
            if not w:
                continue

            # début de mot obligatoire (accepte pluriels / conjugaisons)
            if re.search(r"(?<!\w)" + re.escape(w), text):
                score += 2 + min(len(w.split()), 4)

        scores[label] = score

    if not scores or max(scores.values()) == 0:
        return "À vérifier", 0.0

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best, best_score = ordered[0]
    total = sum(scores.values()) or 1

    confidence = min(99.0, max(35.0, 50 + 50 * best_score / total))

    if len(ordered) > 1 and ordered[1][1] == best_score:
        confidence = min(confidence, 55.0)

    return best, round(confidence, 1)


def ticket_text(titre, description, categorie_glpi=""):
    """La description (nettoyée) est la base du texte analysé."""
    t = clean_text(titre)
    d = clean_text(description)
    c = clean_text(categorie_glpi)
    return f"{t}. {d} {c}".strip()


def build_samples(df, rules):
    """
    Exemples d'apprentissage pour le modèle sémantique :
    - la catégorie corrigée à la main (override) si elle existe
    - sinon la catégorie trouvée par mots-clés (quand elle est sûre)
    """
    samples = []

    for _, r in df.iterrows():
        t = ticket_text(
            r.get("titre", ""),
            r.get("description", ""),
            r.get("categorie_glpi", "")
        )

        ov = norm(r.get("function_override", ""))

        if ov in rules:
            samples.append((t, ov))
            continue

        lab, _ = classify(
            f"{r.get('titre', '')} {r.get('description', '')}",
            rules
        )

        if lab != "À vérifier":
            samples.append((t, lab))

    return samples


@st.cache_resource(show_spinner="Chargement du modèle IA (1ère fois : téléchargement, internet requis)...")
def load_embedder(model_id):
    return SentenceTransformer(model_id)


def embed_batch(texts, rules, samples):
    """
    Compare le SENS du ticket au sens de chaque catégorie
    (nom + description + mots-clés) et aux tickets déjà classés.
    Retourne (catégorie, confiance, méthode) + top 3 dans session_state.
    """
    m = load_embedder(EMBED_MODELS[get_embed_model_name()])
    labels = list(rules)

    ptxt, plab = [], []

    for l in labels:
        words = [w for w in rules[l] if w]
        desc = get_desc(l).strip()

        if desc:
            ptxt.append(desc)
            plab.append(l)
            ptxt.append(f"{l} : {desc}")
            plab.append(l)

        ptxt.append(f"{l} : " + ", ".join(words))
        plab.append(l)

        for w in words:
            ptxt.append(f"{l} : {w}")
            plab.append(l)

    for t, l in samples[:2000]:
        if l in rules and t.strip():
            ptxt.append(t[:600])
            plab.append(l)

    P = m.encode(ptxt, normalize_embeddings=True, batch_size=64)
    Q = m.encode([t[:600] for t in texts], normalize_embeddings=True, batch_size=64)

    sims = Q @ P.T

    idx = {l: [i for i, x in enumerate(plab) if x == l] for l in labels}

    out, tops = [], []

    for row in sims:
        sc = np.array([
            np.sort(row[idx[l]])[-3:].mean() for l in labels
        ])
        e = np.exp((sc - sc.max()) * 20)
        pr = e / e.sum()
        order = np.argsort(-pr)
        i = int(order[0])
        out.append((labels[i], round(float(pr[i]) * 100, 1), "embeddings"))
        tops.append([(labels[j], round(float(pr[j]) * 100, 1)) for j in order[:3]])

    st.session_state["ia_top3"] = tops

    return out


def semantic_batch(texts, rules, samples):
    st.session_state["ia_error"] = ""
    st.session_state["ia_top3"] = []

    if not texts:
        return []

    if not EMBED_AVAILABLE:
        st.session_state["ia_error"] = (
            "sentence-transformers n'est pas installé : "
            "python -m pip install sentence-transformers"
        )
        return [("À vérifier", 0.0, "indisponible")] * len(texts)

    try:
        return embed_batch(texts, rules, samples)
    except Exception as e:
        st.session_state["ia_error"] = f"Embeddings : {e}"
        return [("À vérifier", 0.0, "erreur")] * len(texts)


def classify_rows(df, rules, samples_df=None):
    """
    df : colonnes titre, description, categorie_glpi, function_override.
    1) mots-clés ; 2) pour le reste, embeddings (sens).
    """
    samples = build_samples(samples_df if samples_df is not None else df, rules)

    results = [None] * len(df)
    pending = []

    for pos, (_, r) in enumerate(df.iterrows()):

        lab, conf = classify(f"{r['titre']} {r['description']}", rules)

        if lab != "À vérifier":
            results[pos] = (lab, conf, "mots-clés")
            continue

        t = ticket_text(r["titre"], r["description"], r["categorie_glpi"])

        if norm(t):
            pending.append((pos, t))
        else:
            results[pos] = ("À vérifier", 0.0, "texte vide")

    if pending:
        sem = semantic_batch([t for _, t in pending], rules, samples)
        for (pos, _), r in zip(pending, sem):
            results[pos] = r

    # Plus de « À vérifier » : si rien n'a pu classer le ticket
    # (texte vide, IA indisponible ou en erreur), on prend la catégorie
    # la plus fréquente parmi les tickets déjà classés.
    labels_known = [l for _, l in samples if l in rules]

    if labels_known:
        fallback = max(set(labels_known), key=labels_known.count)
    else:
        fallback = list(rules)[0]

    for pos, r in enumerate(results):
        if r is None or r[0] not in rules:
            conf = r[1] if r else 0.0
            results[pos] = (fallback, conf, "par défaut")

    return results


def reclassify_all():
    """Recalcule la catégorie auto de tous les tickets déjà en base."""
    rules = load_rules()

    with db() as con:
        t = pd.read_sql_query("SELECT * FROM tickets", con)

    if t.empty:
        return 0

    for c in ["titre", "description", "categorie_glpi", "function_override"]:
        t[c] = t[c].fillna("")

    res = classify_rows(t, rules)

    with db() as con:
        for (_, r), (lab, _, _) in zip(t.iterrows(), res):
            con.execute(
                "UPDATE tickets SET function_auto=? WHERE id=?",
                (lab, r["id"])
            )

    return len(t)


# -----------------------------
# GLPI Import
# -----------------------------

GLPI_RENAME = {
    "ID": "id",
    "Titre": "titre",
    "Entité": "entite",
    "Statut": "statut_glpi",
    "Description": "description",
    "Date d'ouverture": "date_ouverture",
    "Priorité": "priorite",
    "Demandeur - Demandeur": "demandeur",
    # Le technicien vient directement de GLPI
    "Attribué à - Technicien": "technicien",
    "Catégorie": "categorie_glpi",
    "Dernière modification": "derniere_modif",
}


def read_glpi(path):
    """
    Lit TOUTES les colonnes du fichier GLPI.
    - les colonnes connues sont mappées vers les champs internes
    - toutes les colonnes (connues ou non) sont aussi gardées brutes dans raw_json
    - une colonne connue absente du fichier devient vide (pas d'erreur)
    """
    raw = pd.read_excel(path)

    # Noms de colonnes propres (espaces parasites)
    raw.columns = [str(c).strip() for c in raw.columns]

    if "ID" not in raw.columns:
        raise ValueError(
            "Colonne « ID » introuvable dans le fichier. "
            "Colonnes trouvées : " + ", ".join(raw.columns)
        )

    # Tout en texte
    rawstr = raw.copy()
    for c in rawstr.columns:
        rawstr[c] = rawstr[c].map(norm)

    rawstr["ID"] = rawstr["ID"].str.replace(r"\.0$", "", regex=True)

    # Champs internes
    df = pd.DataFrame(index=rawstr.index)

    for src, dst in GLPI_RENAME.items():
        df[dst] = rawstr[src] if src in rawstr.columns else ""

    # JSON brut : toutes les colonnes du fichier
    df["raw_json"] = [
        json.dumps(rec, ensure_ascii=False)
        for rec in rawstr.to_dict(orient="records")
    ]

    df = df[df["id"] != ""]

    return df.drop_duplicates("id", keep="last").reset_index(drop=True)


def get_tickets():
    with db() as con:
        return pd.read_sql_query("SELECT * FROM tickets", con)


def expand_raw(df):
    """
    Déplie raw_json en colonnes « Brut · <nom de colonne> ».
    Retourne (df enrichi, liste des noms de colonnes brutes).
    """
    if "raw_json" not in df.columns:
        return df, []

    def parse(s):
        try:
            return json.loads(s) if isinstance(s, str) and s else {}
        except Exception:
            return {}

    recs = [parse(s) for s in df["raw_json"]]

    raw_df = pd.DataFrame(recs, index=df.index).fillna("")

    # On garde l'ordre des colonnes du fichier
    raw_df.columns = [RAW_PREFIX + str(c) for c in raw_df.columns]

    names = list(raw_df.columns)

    out = pd.concat([df, raw_df], axis=1)

    return out, names


def import_glpi(path):
    rules = load_rules()

    incoming = read_glpi(path)

    now = datetime.now().isoformat(timespec="seconds")

    with db() as con:

        existing = pd.read_sql_query("SELECT * FROM tickets", con)

        existing = (
            existing.set_index("id")
            if not existing.empty
            else pd.DataFrame().set_index(pd.Index([], name="id"))
        )

        # --- Modèle sémantique entraîné sur les tickets déjà connus ---
        train_df = incoming.copy()
        train_df["function_override"] = [
            norm(existing.loc[i, "function_override"])
            if i in existing.index else ""
            for i in train_df["id"]
        ]
        auto_results = classify_rows(train_df, rules)

        rows = []

        for pos, (_, r) in enumerate(incoming.iterrows()):

            rid = r["id"]

            old = (
                existing.loc[rid].to_dict()
                if rid in existing.index
                else {}
            )

            fn_auto = auto_results[pos][0]

            rows.append({
                **r.to_dict(),
                "function_auto": fn_auto,
                "function_override": norm(old.get("function_override", "")),
                "action": norm(old.get("action", "")),
                "blocage": norm(old.get("blocage", "")),
                "jira": norm(old.get("jira", "")),
                # Ancien champ conservé uniquement pour compatibilité DB
                "responsable": norm(old.get("responsable", "")),
                "deadline": norm(old.get("deadline", "")),
                "statut_operationnel": norm(old.get("statut_operationnel", "")) or "Nouveau",
                "date_cloture": norm(old.get("date_cloture", "")),
                "commentaire": norm(old.get("commentaire", "")),
                "first_seen": old.get("first_seen", now),
                "last_import": now,
            })

        merged = pd.DataFrame(rows)

        cols = [
            "id", "titre", "entite", "statut_glpi", "description",
            "date_ouverture", "priorite", "demandeur", "technicien",
            "categorie_glpi", "derniere_modif", "function_auto",
            "function_override", "action", "blocage", "jira",
            "responsable", "deadline", "statut_operationnel",
            "date_cloture", "commentaire", "first_seen", "last_import",
            "raw_json"
        ]

        merged = merged[cols]

        merged.to_sql("tickets", con, if_exists="replace", index=False)

        cur = con.execute(
            """
            INSERT INTO snapshots(imported_at, source_file, ticket_count)
            VALUES (?,?,?)
            """,
            (now, os.path.basename(path), len(merged))
        )

        import_id = cur.lastrowid

        snap = merged[
            ["id", "statut_glpi", "categorie_glpi", "function_auto", "technicien"]
        ].copy()

        snap = snap.rename(columns={"id": "ticket_id"})
        snap["import_id"] = import_id

        snap.to_sql("snapshot_rows", con, if_exists="append", index=False)

    return len(merged), import_id


def import_master(master_path):
    """Bootstrap des données manuelles existantes (ancien fichier BKFI AMOA)."""

    if not master_path or not os.path.exists(master_path):
        return 0

    try:
        x = pd.read_excel(master_path, sheet_name="Action_Log")
    except Exception:
        return 0

    if "N° ticket GLPI" not in x.columns:
        return 0

    x["N° ticket GLPI"] = (
        x["N° ticket GLPI"].astype(str).str.replace(r"\.0$", "", regex=True)
    )

    with db() as con:

        count = 0

        for _, r in x.iterrows():

            tid = norm(r.get("N° ticket GLPI"))

            if not tid or tid.lower() in {"nan", "#n/a"}:
                continue

            exists = con.execute(
                "SELECT id FROM tickets WHERE id=?", (tid,)
            ).fetchone()

            if not exists:
                continue

            con.execute(
                """
                UPDATE tickets
                SET
                    function_override=?,
                    action=?,
                    blocage=?,
                    jira=?,
                    deadline=?,
                    statut_operationnel=?,
                    date_cloture=?,
                    commentaire=?
                WHERE id=?
                """,
                (
                    # "Function" = nom de la colonne dans l'ancien fichier Excel
                    norm(r.get("Function")),
                    norm(r.get("Action")),
                    norm(r.get("Blocage")),
                    norm(r.get("N° ticket Jira")),
                    norm(r.get("Deadline")),
                    norm(r.get("Statut opérationnel")),
                    norm(r.get("Date de clôture")),
                    norm(r.get("Commentaire")),
                    tid
                )
            )

            count += 1

    return count


# -----------------------------
# Pivot / TCD
# -----------------------------

def pivot_table(df, rows, cols, value, agg):

    if not rows:
        return pd.DataFrame()

    if value == "Nombre de tickets":
        temp = df.copy()
        temp["__count"] = 1
        value_col = "__count"
        aggfunc = "sum"
    else:
        temp = df
        value_col = value
        aggfunc = {
            "sum": "sum",
            "moyenne": "mean",
            "min": "min",
            "max": "max",
            "count": "count"
        }.get(agg, "count")

    kwargs = dict(
        index=rows,
        values=value_col,
        aggfunc=aggfunc,
        fill_value=0,
        margins=True,
        margins_name="Total"
    )

    if cols:
        kwargs["columns"] = cols

    p = pd.pivot_table(temp, **kwargs)

    return p.reset_index()


# -----------------------------
# UI
# -----------------------------

init_db()

st.markdown("# BKFI AMOA — GLPI Weekly")

st.caption("Plateforme de pilotage des tickets GLPI.")


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.header("⚙️ Données")

    glpi_path = st.file_uploader(
        "Nouvel extract GLPI (.xlsx)",
        type=["xlsx"]
    )

    master_path = st.file_uploader(
        "Ancien BKFI AMOA (optionnel)",
        type=["xlsx"]
    )

    if st.button("🔄 Importer / Actualiser", use_container_width=True):

        if glpi_path is None:

            st.error("Sélectionne d'abord l'extract GLPI du jour.")

        else:

            temp = APP_DIR / "_incoming_glpi.xlsx"

            temp.write_bytes(glpi_path.getvalue())

            try:

                n, iid = import_glpi(temp)

                if master_path is not None:

                    mp = APP_DIR / "_master.xlsx"

                    mp.write_bytes(master_path.getvalue())

                    import_master(mp)

                st.success(
                    f"Import terminé : {n} tickets — snapshot #{iid}."
                )

                st.rerun()

            except Exception as e:

                st.exception(e)

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Action Log",
            "Analyse TCD",
            "Workload",
            "Tickets",
            "Référentiel / IA"
        ],
        index=0
    )


# -----------------------------
# Load tickets
# -----------------------------

df = get_tickets()

if df.empty:

    st.info(
        "Importe ton extract GLPI pour commencer. "
        "Tu peux aussi fournir ton fichier BKFI AMOA "
        "pour récupérer le suivi métier existant."
    )

    st.stop()


# -----------------------------
# Effective fields
# -----------------------------

# Colonnes brutes du fichier GLPI (toutes), préfixées « Brut · »
df, RAW_COLS = expand_raw(df)

# Catégorie (métier) : la correction manuelle prime sur l'automatique.
df["Catégorie"] = df["function_override"].where(
    df["function_override"].fillna("").str.strip() != "",
    df["function_auto"]
)

df["Catégorie GLPI"] = df["categorie_glpi"]

df["Statut GLPI"] = df["statut_glpi"]

df["Priorité"] = df["priorite"]

df["Description"] = df["description"]

# Technicien GLPI ("Attribué à - Technicien"), sinon "Non affecté"
df["Technicien"] = (
    df["technicien"]
    .fillna("")
    .astype(str)
    .str.strip()
    .replace("", "Non affecté")
)

# Sécurité : les champs utilisés par les graphiques ne doivent jamais être NaN
for _c in ["Statut GLPI", "Catégorie GLPI", "Priorité", "Catégorie"]:
    df[_c] = df[_c].fillna("")


# -----------------------------
# Common filters
# -----------------------------

with st.sidebar:

    st.header("🔎 Filtres")

    statuses = sorted([x for x in df["Statut GLPI"].dropna().unique() if x])
    cats = sorted([x for x in df["Catégorie GLPI"].dropna().unique() if x])
    categories = sorted([x for x in df["Catégorie"].dropna().unique() if x])
    technicians = sorted([x for x in df["Technicien"].dropna().unique() if x])

    chosen_status = st.multiselect("Statut GLPI", statuses)

    chosen_categorie = st.multiselect("Catégorie", categories)

    chosen_cat = st.multiselect("Catégorie GLPI", cats)

    chosen_technician = st.multiselect("Technicien", technicians)

    search = st.text_input("🔍 Recherche ticket / titre")


filtered = df.copy()

if chosen_status:
    filtered = filtered[filtered["Statut GLPI"].isin(chosen_status)]

if chosen_categorie:
    filtered = filtered[filtered["Catégorie"].isin(chosen_categorie)]

if chosen_cat:
    filtered = filtered[filtered["Catégorie GLPI"].isin(chosen_cat)]

if chosen_technician:
    filtered = filtered[filtered["Technicien"].isin(chosen_technician)]

if search:

    q = search.lower()

    mask = (
        filtered.astype(str)
        .apply(lambda col: col.str.lower().str.contains(re.escape(q), na=False))
        .any(axis=1)
    )

    filtered = filtered[mask]


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.subheader("Dashboard Weekly")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Total tickets", len(filtered))

    c2.metric(
        "Ouverts / en cours",
        int(
            (
                ~filtered["Statut GLPI"].str.lower().isin(
                    ["résolu", "clos", "closed", "fermé"]
                )
            ).sum()
        )
    )

    c3.metric(
        "Résolus",
        int(filtered["Statut GLPI"].str.lower().eq("résolu").sum())
    )

    c4.metric(
        "Clos",
        int(
            filtered["Statut GLPI"].str.lower().isin(
                ["clos", "fermé", "closed"]
            ).sum()
        )
    )

    c5.metric(
        "Sans technicien",
        int((filtered["Technicien"] == "Non affecté").sum())
    )

    st.divider()

    a, b = st.columns(2)

    with a:
        st.markdown("#### Tickets par Catégorie")
        st.bar_chart(
            filtered["Catégorie"].value_counts().head(12),
            horizontal=True
        )

    with b:
        st.markdown("#### Tickets par statut GLPI")
        st.bar_chart(
            filtered["Statut GLPI"].value_counts(),
            horizontal=True
        )

    a, b = st.columns(2)

    with a:
        st.markdown("#### Catégories GLPI")
        st.bar_chart(
            filtered["Catégorie GLPI"].value_counts().head(15),
            horizontal=True
        )

    with b:
        st.markdown("#### Priorités")
        st.bar_chart(
            filtered["Priorité"].value_counts(),
            horizontal=True
        )


# ============================================================
# ACTION LOG
# ============================================================

elif page == "Action Log":

    st.subheader("Action Log — vue paramétrable")

    display_map = {
        "ID ticket": "id",
        "Titre": "titre",
        "Description": "Description",
        "Catégorie": "Catégorie",
        "Action": "action",
        "Date ouverture": "date_ouverture",
        "Demandeur": "demandeur",
        "Catégorie GLPI": "Catégorie GLPI",
        "Priorité": "Priorité",
        "Blocage": "blocage",
        "Jira": "jira",
        "Technicien": "Technicien",
        "Deadline": "deadline",
        "Statut GLPI": "Statut GLPI",
        "Statut opérationnel": "statut_operationnel",
        "Dernière modification": "derniere_modif",
        "Date clôture": "date_cloture",
        "Commentaire": "commentaire"
    }

    # Toutes les colonnes brutes du fichier GLPI
    for rc in RAW_COLS:
        display_map[rc] = rc

    selected = st.multiselect(
        "Colonnes",
        list(display_map),
        default=[
            "ID ticket",
            "Titre",
            "Description",
            "Catégorie",
            "Action",
            "Catégorie GLPI",
            "Priorité",
            "Technicien",
            "Deadline",
            "Statut GLPI",
            "Statut opérationnel"
        ]
    )

    shown = (
        filtered[[display_map[x] for x in selected]].copy()
        if selected
        else pd.DataFrame()
    )

    st.dataframe(
        shown,
        use_container_width=True,
        height=600,
        hide_index=True
    )

    st.download_button(
        "⬇️ Exporter cette vue Excel-compatible (CSV)",
        shown.to_csv(index=False).encode("utf-8-sig"),
        "action_log_view.csv",
        "text/csv"
    )


# ============================================================
# ANALYSE TCD
# ============================================================

elif page == "Analyse TCD":

    st.subheader("📐 Analyse dynamique — style Tableau Croisé Dynamique")

    fields = [
        "Catégorie",
        "Catégorie GLPI",
        "Statut GLPI",
        "Priorité",
        "Technicien",
        "Description",
        "demandeur",
        "entite",
        "statut_operationnel"
    ] + RAW_COLS

    rows = st.multiselect(
        "Étiquettes de lignes",
        fields,
        default=["Catégorie"]
    )

    cols = st.multiselect(
        "Étiquettes de colonnes",
        fields,
        default=["Statut GLPI"]
    )

    value = st.selectbox("Valeur", ["Nombre de tickets"])

    agg = st.selectbox(
        "Agrégation",
        ["count", "sum", "moyenne", "min", "max"]
    )

    if rows:

        p = pivot_table(filtered, rows, cols, value, agg)

        st.dataframe(
            p,
            use_container_width=True,
            height=550,
            hide_index=True
        )

        st.download_button(
            "⬇️ Exporter le TCD",
            p.to_csv(index=False).encode("utf-8-sig"),
            "tcd.csv",
            "text/csv"
        )

    else:

        st.warning("Choisis au moins un champ dans Étiquettes de lignes.")


# ============================================================
# WORKLOAD
# ============================================================

elif page == "Workload":

    st.subheader("Workload par Technicien")

    wl = (
        filtered.groupby("Technicien")
        .size()
        .sort_values(ascending=False)
        .rename("Tickets")
        .reset_index()
    )

    st.dataframe(wl, use_container_width=True, hide_index=True)

    st.bar_chart(wl.set_index("Technicien"), horizontal=True)


# ============================================================
# TICKETS
# ============================================================

elif page == "Tickets":

    st.subheader("Ticket 360°")

    ids = filtered["id"].tolist()

    if not ids:
        st.info("Aucun ticket avec ces filtres.")
        st.stop()

    tid = st.selectbox("Ticket", ids)

    r = filtered[filtered["id"] == tid].iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Statut GLPI", r["Statut GLPI"])
    c2.metric("Catégorie", r["Catégorie"])
    c3.metric("Priorité", r["Priorité"])
    c4.metric("Technicien", r["Technicien"])

    st.markdown(f"### #{tid} — {r['titre']}")

    st.write(r["description"])

    left, right = st.columns(2)

    with left:
        st.write("**Catégorie GLPI :**", r["Catégorie GLPI"])
        st.write("**Catégorie automatique :**", r["function_auto"])
        st.write("**Catégorie retenue :**", r["Catégorie"])
        st.write("**Demandeur :**", r["demandeur"])
        st.write("**Technicien GLPI :**", r["Technicien"])

    with right:
        st.write("**Action :**", r["action"])
        st.write("**Blocage :**", r["blocage"])
        st.write("**Jira :**", r["jira"])
        st.write("**Deadline :**", r["deadline"])
        st.write("**Statut opérationnel :**", r["statut_operationnel"])

    with st.expander("📄 Toutes les colonnes du fichier GLPI (brut)"):
        raw_view = pd.DataFrame({
            "Colonne": [c.replace(RAW_PREFIX, "", 1) for c in RAW_COLS],
            "Valeur": [r[c] for c in RAW_COLS],
        })
        st.dataframe(raw_view, use_container_width=True, hide_index=True)


# ============================================================
# REFERENTIEL / IA
# ============================================================

elif page == "Référentiel / IA":

    st.subheader("🧠 Référentiel & classification automatique")

    rules = load_rules()

    st.write(
        "Chaque ticket reçoit une **Catégorie** parmi : "
        + ", ".join(f"**{k}**" for k in rules)
        + ". Si aucun mot-clé ne correspond, le système analyse le sens "
        "du texte (titre + description + catégorie GLPI) et choisit "
        "automatiquement la catégorie la plus proche, en "
        "s'appuyant sur les tickets déjà classés et sur les corrections manuelles."
    )

    st.markdown("### Moteur IA — Embeddings local (pour les tickets sans mot-clé)")

    st.caption("Comprend le sens du texte. Tourne en local : les données ne sortent pas du PC.")

    if not EMBED_AVAILABLE:
        st.warning(
            "sentence-transformers n'est pas installé : "
            "`python -m pip install sentence-transformers`"
        )

    names = list(EMBED_MODELS)
    cur_model = get_embed_model_name()

    chosen_model = st.radio(
        "Modèle",
        names,
        index=names.index(cur_model),
        horizontal=True
    )

    if chosen_model != cur_model:
        set_setting("embed_model", chosen_model)
        st.info("Modèle changé : clique sur « Reclasser tous les tickets existants ».")

    with st.expander("📝 Description de chaque catégorie (améliore beaucoup la précision)"):
        st.caption(
            "Écris une phrase simple qui explique ce que contient la catégorie. "
            "L'IA compare le ticket à ces phrases."
        )
        for l in rules:
            d = st.text_area(
                "Description — " + l,
                get_desc(l),
                key="desc_" + l,
                height=70
            )
            if d != get_desc(l):
                set_setting("desc:" + l, d)

    if st.session_state.get("ia_error"):
        st.error(st.session_state["ia_error"])

    labels = list(rules)

    for label in labels:

        text = st.text_area(
            label,
            ", ".join(rules[label]),
            key="rule_" + label
        )

        rules[label] = [
            x.strip().lower()
            for x in text.split(",")
            if x.strip()
        ]

    b1, b2 = st.columns(2)

    if b1.button("💾 Enregistrer le référentiel", type="primary"):

        save_rules(rules)

        st.success(
            "Référentiel enregistré. "
            "Les nouveaux imports utiliseront ces règles."
        )

    if b2.button("♻️ Reclasser tous les tickets existants"):

        save_rules(rules)

        with st.spinner("Classification en cours..."):
            n = reclassify_all()

        st.success(f"{n} tickets reclassés.")

        if st.session_state.get("ia_error"):
            st.error(st.session_state["ia_error"])

    st.divider()

    st.markdown("### Testeur de classification")

    txt = st.text_area("Texte du ticket", "")

    if txt:

        train = df.copy()
        for c in ["titre", "description", "categorie_glpi", "function_override"]:
            train[c] = train[c].fillna("")

        tdf = pd.DataFrame([{
            "titre": txt, "description": "",
            "categorie_glpi": "", "function_override": ""
        }])

        lab, conf, method = classify_rows(
            tdf, rules, samples_df=train
        )[0]

        if st.session_state.get("ia_error"):
            st.error(st.session_state["ia_error"])

        st.metric(
            "Catégorie proposée",
            lab,
            f"Confiance {conf}% — méthode : {method}"
        )

        top3 = st.session_state.get("ia_top3") or []
        if top3 and method == "embeddings":
            st.caption(
                "Top 3 : " + " · ".join(f"{l} ({c}%)" for l, c in top3[0])
            )

        if conf < 60:

            st.warning(
                "Confiance faible : vérification humaine recommandée."
            )


# -----------------------------
# Footer
# -----------------------------

st.caption("Données affichées depuis Extraction GLPI")
