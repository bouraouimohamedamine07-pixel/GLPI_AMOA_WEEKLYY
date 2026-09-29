import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

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
            responsable TEXT
        )""")

        con.execute("""CREATE TABLE IF NOT EXISTS rules (
            label TEXT PRIMARY KEY,
            keywords TEXT
        )""")


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

    return {
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


def classify(text, rules):
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
        max(
            35.0,
            50 + 50 * best_score / total
        )
    )

    if len(ordered) > 1 and ordered[1][1] == best_score:
        confidence = min(confidence, 55.0)

    return best, round(confidence, 1)


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

        # IMPORTANT :
        # Le technicien vient directement de GLPI
        "Attribué à - Technicien": "technicien",

        "Catégorie": "categorie_glpi",
        "Dernière modification": "derniere_modif",
    }

    missing = [
        x for x in rename
        if x not in raw.columns
    ]

    if missing:
        raise ValueError(
            "Colonnes GLPI manquantes: "
            + ", ".join(missing)
        )

    df = raw.rename(columns=rename)[
        list(rename.values())
    ].copy()

    df["id"] = (
        df["id"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )

    for c in df.columns:
        df[c] = df[c].map(norm)

    return df.drop_duplicates(
        "id",
        keep="last"
    )


def get_tickets():
    with db() as con:
        return pd.read_sql_query(
            "SELECT * FROM tickets",
            con
        )


def import_glpi(path):
    rules = load_rules()

    incoming = read_glpi(path)

    now = datetime.now().isoformat(
        timespec="seconds"
    )

    with db() as con:

        existing = pd.read_sql_query(
            "SELECT * FROM tickets",
            con
        )

        existing = (
            existing.set_index("id")
            if not existing.empty
            else pd.DataFrame().set_index(
                pd.Index([], name="id")
            )
        )

        rows = []

        for _, r in incoming.iterrows():

            rid = r["id"]

            old = (
                existing.loc[rid].to_dict()
                if rid in existing.index
                else {}
            )

            fn_auto, _ = classify(
                f"{r['titre']} {r['description']}",
                rules
            )

            rows.append({
                **r.to_dict(),

                "function_auto": fn_auto,

                "function_override":
                    old.get(
                        "function_override",
                        ""
                    ),

                "action":
                    old.get("action", ""),

                "blocage":
                    old.get("blocage", ""),

                "jira":
                    old.get("jira", ""),

                # Ancien champ conservé uniquement
                # pour compatibilité avec la DB.
                # Il n'est plus utilisé dans le dashboard.
                "responsable":
                    old.get("responsable", ""),

                "deadline":
                    old.get("deadline", ""),

                "statut_operationnel":
                    old.get(
                        "statut_operationnel",
                        "Nouveau"
                    ),

                "date_cloture":
                    old.get("date_cloture", ""),

                "commentaire":
                    old.get("commentaire", ""),

                "first_seen":
                    old.get("first_seen", now),

                "last_import":
                    now,
            })

        merged = pd.DataFrame(rows)

        cols = [
            "id",
            "titre",
            "entite",
            "statut_glpi",
            "description",
            "date_ouverture",
            "priorite",
            "demandeur",
            "technicien",
            "categorie_glpi",
            "derniere_modif",
            "function_auto",
            "function_override",
            "action",
            "blocage",
            "jira",
            "responsable",
            "deadline",
            "statut_operationnel",
            "date_cloture",
            "commentaire",
            "first_seen",
            "last_import"
        ]

        merged = merged[cols]

        merged.to_sql(
            "tickets",
            con,
            if_exists="replace",
            index=False
        )

        cur = con.execute(
            """
            INSERT INTO snapshots(
                imported_at,
                source_file,
                ticket_count
            )
            VALUES (?,?,?)
            """,
            (
                now,
                os.path.basename(path),
                len(merged)
            )
        )

        import_id = cur.lastrowid

        snap = merged[
            [
                "id",
                "statut_glpi",
                "categorie_glpi",
                "function_auto"
            ]
        ].copy()

        # Le snapshot conserve désormais le technicien GLPI
        snap["technicien"] = merged["technicien"]
        snap["import_id"] = import_id

        snap.to_sql(
            "snapshot_rows",
            con,
            if_exists="append",
            index=False
        )

    return len(merged), import_id


def import_master(master_path):
    """
    Bootstrap des données manuelles existantes.

    IMPORTANT :
    Le champ Responsable de l'ancien fichier n'est plus
    utilisé comme responsable dans l'application.

    Le Technicien reste celui provenant directement de GLPI.
    """

    if not master_path or not os.path.exists(master_path):
        return 0

    try:
        x = pd.read_excel(
            master_path,
            sheet_name="Action_Log"
        )
    except Exception:
        return 0

    if "N° ticket GLPI" not in x.columns:
        return 0

    x["N° ticket GLPI"] = (
        x["N° ticket GLPI"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )

    with db() as con:

        count = 0

        for _, r in x.iterrows():

            tid = norm(
                r.get("N° ticket GLPI")
            )

            if not tid or tid.lower() in {
                "nan",
                "#n/a"
            }:
                continue

            exists = con.execute(
                "SELECT id FROM tickets WHERE id=?",
                (tid,)
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

        value_col = value

        aggfunc = {
            "sum": "sum",
            "moyenne": "mean",
            "min": "min",
            "max": "max",
            "count": "count"
        }.get(
            agg,
            "count"
        )

    if cols:

        p = pd.pivot_table(
            temp if value == "Nombre de tickets" else df,
            index=rows,
            columns=cols,
            values=value_col,
            aggfunc=aggfunc,
            fill_value=0,
            margins=True,
            margins_name="Total"
        )

    else:

        p = pd.pivot_table(
            temp if value == "Nombre de tickets" else df,
            index=rows,
            values=value_col,
            aggfunc=aggfunc,
            fill_value=0,
            margins=True,
            margins_name="Total"
        )

    return p.reset_index()


# -----------------------------
# UI
# -----------------------------

init_db()

st.markdown("# BKFI AMOA — GLPI Weekly")

st.caption(
    "Plateforme de pilotage des tickets GLPI."
)


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

    if st.button(
        "🔄 Importer / Actualiser",
        use_container_width=True
    ):

        if glpi_path is None:

            st.error(
                "Sélectionne d'abord l'extract GLPI du jour."
            )

        else:

            temp = APP_DIR / "_incoming_glpi.xlsx"

            temp.write_bytes(
                glpi_path.getvalue()
            )

            try:

                n, iid = import_glpi(temp)

                if master_path is not None:

                    mp = APP_DIR / "_master.xlsx"

                    mp.write_bytes(
                        master_path.getvalue()
                    )

                    import_master(mp)

                st.success(
                    f"Import terminé : {n} tickets — "
                    f"snapshot #{iid}."
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

# Effective Function:
# manual override wins.
df["Function"] = df["function_override"].where(
    df["function_override"]
    .fillna("")
    .str.strip() != "",
    df["function_auto"]
)

df["Catégorie GLPI"] = df["categorie_glpi"]

df["Statut GLPI"] = df["statut_glpi"]

df["Priorité"] = df["priorite"]


# IMPORTANT
# Le Responsable est maintenant remplacé par le Technicien.
#
# Source :
# GLPI -> "Attribué à - Technicien"
#
# Si le ticket n'a pas de technicien :
# "Non affecté"

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

    statuses = sorted([
        x
        for x in df["Statut GLPI"].dropna().unique()
        if x
    ])

    cats = sorted([
        x
        for x in df["Catégorie GLPI"].dropna().unique()
        if x
    ])

    funcs = sorted([
        x
        for x in df["Function"].dropna().unique()
        if x
    ])

    technicians = sorted([
        x
        for x in df["Technicien"].dropna().unique()
        if x
    ])

    chosen_status = st.multiselect(
        "Statut GLPI",
        statuses
    )

    chosen_func = st.multiselect(
        "Function",
        funcs
    )

    chosen_cat = st.multiselect(
        "Catégorie GLPI",
        cats
    )

    chosen_technician = st.multiselect(
        "Technicien",
        technicians
    )

    search = st.text_input(
        "🔍 Recherche ticket / titre"
    )


filtered = df.copy()

if chosen_status:

    filtered = filtered[
        filtered["Statut GLPI"].isin(
            chosen_status
        )
    ]

if chosen_func:

    filtered = filtered[
        filtered["Function"].isin(
            chosen_func
        )
    ]

if chosen_cat:

    filtered = filtered[
        filtered["Catégorie GLPI"].isin(
            chosen_cat
        )
    ]

if chosen_technician:

    filtered = filtered[
        filtered["Technicien"].isin(
            chosen_technician
        )
    ]

if search:

    q = search.lower()

    mask = (
        filtered
        .astype(str)
        .apply(
            lambda col:
            col.str.lower()
            .str.contains(
                re.escape(q),
                na=False
            )
        )
        .any(axis=1)
    )

    filtered = filtered[mask]


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.subheader("Dashboard Weekly")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Total tickets",
        len(filtered)
    )

    c2.metric(
        "Ouverts / en cours",
        int(
            (
                ~filtered["Statut GLPI"]
                .str.lower()
                .isin([
                    "résolu",
                    "clos",
                    "closed",
                    "fermé"
                ])
            ).sum()
        )
    )

    c3.metric(
        "Résolus",
        int(
            filtered["Statut GLPI"]
            .str.lower()
            .eq("résolu")
            .sum()
        )
    )

    c4.metric(
        "Clos",
        int(
            filtered["Statut GLPI"]
            .str.lower()
            .isin([
                "clos",
                "fermé",
                "closed"
            ])
            .sum()
        )
    )

    c5.metric(
        "Sans technicien",
        int(
            (
                filtered["Technicien"]
                == "Non affecté"
            ).sum()
        )
    )

    st.divider()

    a, b = st.columns(2)

    with a:

        st.markdown(
            "#### Tickets par Function"
        )

        st.bar_chart(
            filtered["Function"]
            .value_counts()
            .head(12)
            horizontal=True
        )

    with b:

        st.markdown(
            "#### Tickets par statut GLPI"
        )

        st.bar_chart(
            filtered["Statut GLPI"]
            .value_counts()
        )

    a, b = st.columns(2)

    with a:

        st.markdown(
            "#### Catégories GLPI"
        )

        st.bar_chart(
            filtered["Catégorie GLPI"]
            .value_counts()
            .head(15)
        )

    with b:

        st.markdown(
            "#### Priorités"
        )

        st.bar_chart(
            filtered["Priorité"]
            .value_counts()
        )


# ============================================================
# ACTION LOG
# ============================================================

elif page == "Action Log":

    st.subheader(
        "Action Log — vue paramétrable"
    )

    display_map = {

        "ID ticket":
            "id",

        "Titre":
            "titre",

        "Function":
            "Function",

        "Action":
            "action",

        "Date ouverture":
            "date_ouverture",

        "Demandeur":
            "demandeur",

        "Catégorie GLPI":
            "Catégorie GLPI",

        "Priorité":
            "Priorité",

        "Blocage":
            "blocage",

        "Jira":
            "jira",

        # Responsable remplacé par Technicien
        "Technicien":
            "Technicien",

        "Deadline":
            "deadline",

        "Statut GLPI":
            "Statut GLPI",

        "Statut opérationnel":
            "statut_operationnel",

        "Dernière modification":
            "derniere_modif",

        "Date clôture":
            "date_cloture",

        "Commentaire":
            "commentaire"
    }

    selected = st.multiselect(
        "Colonnes",
        list(display_map),
        default=[
            "ID ticket",
            "Titre",
            "Function",
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
        filtered[
            [
                display_map[x]
                for x in selected
            ]
        ].copy()
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
        shown.to_csv(
            index=False
        ).encode("utf-8-sig"),
        "action_log_view.csv",
        "text/csv"
    )


# ============================================================
# ANALYSE TCD
# ============================================================

elif page == "Analyse TCD":

    st.subheader(
        "📐 Analyse dynamique — style Tableau Croisé Dynamique"
    )

    fields = [
        "Function",
        "Catégorie GLPI",
        "Statut GLPI",
        "Priorité",

        # Responsable remplacé par Technicien
        "Technicien",

        "demandeur",
        "entite",
        "statut_operationnel"
    ]

    rows = st.multiselect(
        "Étiquettes de lignes",
        fields,
        default=["Function"]
    )

    cols = st.multiselect(
        "Étiquettes de colonnes",
        fields,
        default=["Statut GLPI"]
    )

    value = st.selectbox(
        "Valeur",
        ["Nombre de tickets"]
    )

    agg = st.selectbox(
        "Agrégation",
        [
            "count",
            "sum",
            "moyenne",
            "min",
            "max"
        ]
    )

    if rows:

        p = pivot_table(
            filtered,
            rows,
            cols,
            value,
            agg
        )

        st.dataframe(
            p,
            use_container_width=True,
            height=550,
            hide_index=True
        )

        st.download_button(
            "⬇️ Exporter le TCD",
            p.to_csv(
                index=False
            ).encode("utf-8-sig"),
            "tcd.csv",
            "text/csv"
        )

    else:

        st.warning(
            "Choisis au moins un champ dans "
            "Étiquettes de lignes."
        )


# ============================================================
# WORKLOAD
# ============================================================

elif page == "Workload":

    st.subheader("Workload par Technicien")

    # IMPORTANT :
    # Le workload est maintenant calculé directement
    # depuis Attribué à - Technicien de GLPI.

    wl = (
        filtered
        .groupby("Technicien")
        .size()
        .sort_values(
            ascending=False
        )
        .rename("Tickets")
        .reset_index()
    )

    st.dataframe(
        wl,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        wl.set_index("Technicien")
    )


# ============================================================
# TICKETS
# ============================================================

elif page == "Tickets":

    st.subheader("Ticket 360°")

    ids = filtered["id"].tolist()

    tid = st.selectbox(
        "Ticket",
        ids
    )

    r = filtered[
        filtered["id"] == tid
    ].iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Statut GLPI",
        r["Statut GLPI"]
    )

    c2.metric(
        "Function",
        r["Function"]
    )

    c3.metric(
        "Priorité",
        r["Priorité"]
    )

    c4.metric(
        "Technicien",
        r["Technicien"]
    )

    st.markdown(
        f"### #{tid} — {r['titre']}"
    )

    st.write(
        r["description"]
    )

    left, right = st.columns(2)

    with left:

        st.write(
            "**Catégorie GLPI :**",
            r["Catégorie GLPI"]
        )

        st.write(
            "**Function automatique :**",
            r["function_auto"]
        )

        st.write(
            "**Function retenue :**",
            r["Function"]
        )

        st.write(
            "**Demandeur :**",
            r["demandeur"]
        )

        st.write(
            "**Technicien GLPI :**",
            r["Technicien"]
        )

    with right:

        st.write(
            "**Action :**",
            r["action"]
        )

        st.write(
            "**Blocage :**",
            r["blocage"]
        )

        st.write(
            "**Jira :**",
            r["jira"]
        )

        st.write(
            "**Deadline :**",
            r["deadline"]
        )

        st.write(
            "**Statut opérationnel :**",
            r["statut_operationnel"]
        )


# ============================================================
# REFERENTIEL / IA
# ============================================================

elif page == "Référentiel / IA":

    st.subheader(
        "🧠 Référentiel & classification automatique"
    )

    rules = load_rules()

    st.write(
        "Les catégories métier ci-dessous sont celles "
        " **Anomalie, "
        "Amélioration, Nouvelle demande, "
        "Réglementaire / conformité, Technique, "
        "Data & reporting, Formation, Correction, Accès**. "
    )

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

    if st.button(
        "💾 Enregistrer le référentiel",
        type="primary"
    ):

        save_rules(rules)

        st.success(
            "Référentiel enregistré. "
            "Les nouveaux imports utiliseront ces règles."
        )

    st.divider()

    st.markdown(
        "### Testeur de classification"
    )

    txt = st.text_area(
        "Texte du ticket",
        ""
    )

    if txt:

        lab, conf = classify(
            txt,
            rules
        )

        st.metric(
            "Catégorie métier proposée",
            lab,
            f"Confiance {conf}%"
        )

        if conf < 60:

            st.warning(
                "Confiance faible : "
                "vérification humaine recommandée."
            )


# -----------------------------
# Footer
# -----------------------------

st.caption(
    "Données affichées depuis la base locale du dashboard. "
    "Le champ Technicien provient directement de "
    "« Attribué à - Technicien » dans l'extract GLPI. "
    "Les champs métier Action/Deadline/etc. sont conservés "
    "lors des imports grâce à l'ID GLPI."
)
