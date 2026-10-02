import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# Classification sémantique (optionnelle mais recommandée) : pip install scikit-learn
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

import numpy as np

# Embeddings locaux (comprend le sens) : pip install sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    EMBED_AVAILABLE = True
except Exception:
    EMBED_AVAILABLE = False

# API Claude (optionnel) : pip install anthropic
try:
    import anthropic
    CLAUDE_AVAILABLE = True
except Exception:
    CLAUDE_AVAILABLE = False

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
ENGINES = ["Embeddings local", "Claude API", "Statistique (TF-IDF)"]

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "glpi_dashboard.db"
DEFAULT_MASTER = APP_DIR / "BKFI_AMOA_suivi GLPI.xlsx"

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
            last_import TEXT
        )""")

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


def get_engine():
    e = get_setting("engine", ENGINES[0])
    return e if e in ENGINES else ENGINES[0]


def api_key():
    k = st.session_state.get("api_key", "").strip()
    if k:
        return k
    k = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if k:
        return k
    try:
        return str(st.secrets.get("ANTHROPIC_API_KEY", "")).strip()
    except Exception:
        return ""


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
    if pd.isna(v):
        return ""
    return str(v).strip()


# -----------------------------
# Classification
# -----------------------------

def classify(text, rules):
    """Étape 1 : classification par mots-clés."""
    text = norm(text).lower()

    if not text:
        return "À vérifier", 0.0

    scores = {}

    for label, words in rules.items():
        score = 0

        for w in words:
            if not w:
                continue

            if w in text:
                score += 2 + min(len(w.split()), 4)

        scores[label] = score

    if not scores or max(scores.values()) == 0:
        return "À vérifier", 0.0

    ordered = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    best, best_score = ordered[0]

    total = sum(scores.values()) or 1

    confidence = min(
        99.0,
        max(35.0, 50 + 50 * best_score / total)
    )

    if len(ordered) > 1 and ordered[1][1] == best_score:
        confidence = min(confidence, 55.0)

    return best, round(confidence, 1)


def ticket_text(titre, description, categorie_glpi=""):
    # Le titre est répété pour lui donner plus de poids
    return f"{norm(titre)} {norm(titre)} {norm(description)} {norm(categorie_glpi)}"


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


def build_model(rules, samples):
    """
    TF-IDF sur n-grammes de caractères + régression logistique.
    (moteur "Statistique", le moins précis)
    """
    if not SEMANTIC_AVAILABLE:
        return None

    X, y = [], []

    for label, words in rules.items():
        words = [w for w in words if w]
        for w in words:
            X.append(w)
            y.append(label)
        if words:
            X.append(" ".join(words))
            y.append(label)

    for t, l in samples:
        if t.strip() and l in rules:
            X.append(t)
            y.append(l)

    if len(set(y)) < 2:
        return None

    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True
    )

    Xv = vec.fit_transform(X)

    clf = LogisticRegression(
        max_iter=2000,
        C=5.0,
        class_weight="balanced"
    )

    clf.fit(Xv, y)

    return vec, clf


def predict_semantic(text, model):
    vec, clf = model
    probs = clf.predict_proba(vec.transform([text]))[0]
    i = probs.argmax()
    return str(clf.classes_[i]), round(float(probs[i]) * 100, 1)


@st.cache_resource(show_spinner="Chargement du modèle IA (1ère fois : téléchargement ~120 MB)...")
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)


def embed_batch(texts, rules, samples):
    """
    Moteur "Embeddings local" : compare le SENS du ticket au sens
    de chaque catégorie (nom + mots-clés) et aux tickets déjà classés.
    """
    m = load_embedder()
    labels = list(rules)

    ptxt, plab = [], []

    for l in labels:
        words = [w for w in rules[l] if w]
        ptxt.append(f"{l}: " + ", ".join(words))
        plab.append(l)
        for w in words:
            ptxt.append(f"{l}: {w}")
            plab.append(l)

    for t, l in samples[:2000]:
        if l in rules and t.strip():
            ptxt.append(t[:500])
            plab.append(l)

    P = m.encode(ptxt, normalize_embeddings=True, batch_size=64)
    Q = m.encode([t[:500] for t in texts], normalize_embeddings=True, batch_size=64)

    sims = Q @ P.T

    idx = {l: [i for i, x in enumerate(plab) if x == l] for l in labels}

    out = []

    for row in sims:
        sc = np.array([
            np.sort(row[idx[l]])[-3:].mean() for l in labels
        ])
        e = np.exp((sc - sc.max()) * 20)
        pr = e / e.sum()
        i = int(pr.argmax())
        out.append((labels[i], round(float(pr[i]) * 100, 1), "embeddings"))

    return out


def claude_batch(texts, rules):
    """
    Moteur 2 : API Claude. Choisit la catégorie la plus proche
    parmi celles du référentiel (par lots de 15 tickets).
    """
    import json

    client = anthropic.Anthropic(api_key=api_key())

    labels = list(rules)

    desc = "\n".join(
        f"- {l} (exemples de mots-clés : {', '.join(rules[l][:12])})"
        for l in labels
    )

    out = [None] * len(texts)

    for s0 in range(0, len(texts), 15):

        chunk = texts[s0:s0 + 15]

        items = "\n".join(
            f"[{i}] {t[:600]}" for i, t in enumerate(chunk)
        )

        prompt = (
            "Tu classes des tickets de support GLPI (en français, parfois en arabe "
            "ou en anglais) dans UNE catégorie métier.\n\n"
            f"Catégories possibles :\n{desc}\n\n"
            "Règles : choisis TOUJOURS la catégorie la plus proche dans cette liste, "
            "exactement avec son nom, même si le ticket ne contient aucun mot-clé. "
            "Réponds UNIQUEMENT par un JSON, sans texte autour, de la forme :\n"
            '[{"i": 0, "categorie": "...", "confiance": 0-100}]\n\n'
            f"Tickets :\n{items}"
        )

        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = re.sub(r"```json|```", "", resp.content[0].text).strip()

        try:
            data = json.loads(raw)
        except Exception:
            data = []

        for d in data:
            i = d.get("i")
            lab = d.get("categorie")
            if isinstance(i, int) and 0 <= i < len(chunk) and lab in rules:
                try:
                    c = float(d.get("confiance", 70))
                except Exception:
                    c = 70.0
                out[s0 + i] = (lab, c, "Claude")

    return out


def semantic_batch(texts, rules, samples, engine):
    """Classe des textes sans mot-clé avec le moteur choisi. Retombe sur TF-IDF si besoin."""
    st.session_state["ia_error"] = ""

    if not texts:
        return []

    res = [None] * len(texts)

    try:
        if engine == "Claude API" and CLAUDE_AVAILABLE and api_key():
            res = claude_batch(texts, rules)
        elif engine == "Embeddings local" and EMBED_AVAILABLE:
            res = embed_batch(texts, rules, samples)
    except Exception as e:
        st.session_state["ia_error"] = f"{engine} : {e}"
        res = [None] * len(texts)

    if any(r is None for r in res) and SEMANTIC_AVAILABLE:
        model = build_model(rules, samples)
        if model is not None:
            for i, r in enumerate(res):
                if r is None:
                    l, c = predict_semantic(texts[i], model)
                    res[i] = (l, c, "statistique")

    return [r if r else ("À vérifier", 0.0, "indisponible") for r in res]


def classify_rows(df, rules, engine, samples_df=None):
    """
    df : colonnes titre, description, categorie_glpi, function_override.
    1) mots-clés ; 2) pour le reste, moteur IA.
    Retourne une liste de (catégorie, confiance, méthode).
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
        sem = semantic_batch([t for _, t in pending], rules, samples, engine)
        for (pos, _), r in zip(pending, sem):
            results[pos] = r

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

    res = classify_rows(t, rules, get_engine())

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

def read_glpi(path):
    raw = pd.read_excel(path)

    rename = {
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

    missing = [x for x in rename if x not in raw.columns]

    if missing:
        raise ValueError(
            "Colonnes GLPI manquantes: " + ", ".join(missing)
        )

    df = raw.rename(columns=rename)[list(rename.values())].copy()

    df["id"] = (
        df["id"].astype(str).str.replace(r"\.0$", "", regex=True)
    )

    for c in df.columns:
        df[c] = df[c].map(norm)

    return df.drop_duplicates("id", keep="last")


def get_tickets():
    with db() as con:
        return pd.read_sql_query("SELECT * FROM tickets", con)


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
        auto_results = classify_rows(train_df, rules, get_engine())

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
                "function_override": old.get("function_override", ""),
                "action": old.get("action", ""),
                "blocage": old.get("blocage", ""),
                "jira": old.get("jira", ""),
                # Ancien champ conservé uniquement pour compatibilité DB
                "responsable": old.get("responsable", ""),
                "deadline": old.get("deadline", ""),
                "statut_operationnel": old.get("statut_operationnel", "Nouveau"),
                "date_cloture": old.get("date_cloture", ""),
                "commentaire": old.get("commentaire", ""),
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
            "date_cloture", "commentaire", "first_seen", "last_import"
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

# Catégorie (métier) : la correction manuelle prime sur l'automatique.
df["Catégorie"] = df["function_override"].where(
    df["function_override"].fillna("").str.strip() != "",
    df["function_auto"]
)

df["Catégorie GLPI"] = df["categorie_glpi"]

df["Statut GLPI"] = df["statut_glpi"]

df["Priorité"] = df["priorite"]

# Technicien GLPI ("Attribué à - Technicien"), sinon "Non affecté"
df["Technicien"] = (
    df["technicien"]
    .fillna("")
    .astype(str)
    .str.strip()
    .replace("", "Non affecté")
)


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
        st.bar_chart(filtered["Statut GLPI"].value_counts())

    a, b = st.columns(2)

    with a:
        st.markdown("#### Catégories GLPI")
        st.bar_chart(filtered["Catégorie GLPI"].value_counts().head(15))

    with b:
        st.markdown("#### Priorités")
        st.bar_chart(filtered["Priorité"].value_counts())


# ============================================================
# ACTION LOG
# ============================================================

elif page == "Action Log":

    st.subheader("Action Log — vue paramétrable")

    display_map = {
        "ID ticket": "id",
        "Titre": "titre",
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

    selected = st.multiselect(
        "Colonnes",
        list(display_map),
        default=[
            "ID ticket",
            "Titre",
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
        "demandeur",
        "entite",
        "statut_operationnel"
    ]

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

    st.bar_chart(wl.set_index("Technicien"))


# ============================================================
# TICKETS
# ============================================================

elif page == "Tickets":

    st.subheader("Ticket 360°")

    ids = filtered["id"].tolist()

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

    st.markdown("### Moteur IA (pour les tickets sans mot-clé)")

    cur = get_engine()

    engine = st.radio(
        "Moteur",
        ENGINES,
        index=ENGINES.index(cur),
        horizontal=True
    )

    if engine != cur:
        set_setting("engine", engine)

    if engine == "Embeddings local":
        st.caption("Comprend le sens, tourne en local (données non envoyées).")
        if not EMBED_AVAILABLE:
            st.warning(
                "sentence-transformers n'est pas installé : retombe sur TF-IDF. "
                "`python -m pip install sentence-transformers`"
            )

    elif engine == "Claude API":
        st.warning(
            "⚠️ Le titre et la description des tickets sans mot-clé sont "
            "envoyés à l'API Anthropic. Vérifie que c'est autorisé."
        )
        if not CLAUDE_AVAILABLE:
            st.warning("`python -m pip install anthropic`")
        st.session_state["api_key"] = st.text_input(
            "Clé API Anthropic (gardée en session seulement)",
            type="password",
            value=st.session_state.get("api_key", "")
        )

    else:
        st.caption("Statistique sur les lettres : le moins précis.")
        if not SEMANTIC_AVAILABLE:
            st.warning("`python -m pip install scikit-learn`")

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
            tdf, rules, get_engine(), samples_df=train
        )[0]

        if st.session_state.get("ia_error"):
            st.error(st.session_state["ia_error"])

        st.metric(
            "Catégorie proposée",
            lab,
            f"Confiance {conf}% — méthode : {method}"
        )

        if conf < 60:

            st.warning(
                "Confiance faible : vérification humaine recommandée."
            )


# -----------------------------
# Footer
# -----------------------------

st.caption("Données affichées depuis Extraction GLPI")
