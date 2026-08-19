from pathlib import Path

import streamlit as st

from config import (
    APP_NAME,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DB_PATH,
    EMBEDDING_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    SEMANTIC_BREAKPOINT_PERCENTILE,
    SEMANTIC_CONTEXT_WINDOW,
    SEMANTIC_MIN_BREAKPOINT_DISTANCE,
    SEMANTIC_MIN_CHUNK_SIZE,
    TOP_K,
    VECTOR_DB_DIR,
)
from chunking import semantic_chunk_text
from database import WorkMindDB
from embeddings import get_embedder
from openrouter_client import answer_with_openrouter
from text_utils import iter_text_files, read_text_file, utc_now
from vector_store import VectorStore


st.set_page_config(page_title=APP_NAME, page_icon="WM", layout="wide")


@st.cache_resource
def get_db() -> WorkMindDB:
    return WorkMindDB(DB_PATH)


@st.cache_resource
def get_vector_store() -> VectorStore:
    return VectorStore(VECTOR_DB_DIR)


@st.cache_resource
def load_embedder(model_name: str, api_key: str = ""):
    return get_embedder(model_name, api_key=api_key, base_url=OPENROUTER_BASE_URL)


db = get_db()

if "messages" not in st.session_state:
    st.session_state.messages = []


def project_options(projects: list[dict]) -> dict[str, int]:
    return {f"{p['name']}": int(p["id"]) for p in projects}


def index_documents(
    project_id: int,
    docs: list[dict],
    model_name: str,
    api_key: str = "",
) -> int:
    embedder = load_embedder(model_name, api_key=api_key)
    vector_store = get_vector_store()
    indexed_chunks = 0
    progress = st.progress(0)

    for index, doc in enumerate(docs, start=1):
        chunks = semantic_chunk_text(
            doc["content"],
            embedder,
            max_chars=CHUNK_SIZE,
            overlap_chars=CHUNK_OVERLAP,
            min_chars=SEMANTIC_MIN_CHUNK_SIZE,
            breakpoint_percentile=SEMANTIC_BREAKPOINT_PERCENTILE,
            min_breakpoint_distance=SEMANTIC_MIN_BREAKPOINT_DISTANCE,
            context_window=SEMANTIC_CONTEXT_WINDOW,
        )
        embeddings = embedder.encode(chunks) if chunks else []
        indexed_chunks += vector_store.upsert_chunks(
            project_id,
            doc["path"],
            doc["name"],
            chunks,
            embeddings,
        )
        db.upsert_file(
            project_id,
            doc["name"],
            doc["path"],
            doc["content"],
            doc.get("last_modified") or utc_now(),
        )
        progress.progress(index / len(docs))

    progress.empty()
    return indexed_chunks


with st.sidebar:
    st.title(APP_NAME)
    st.caption("Personal assistant over your indexed files")

    page = st.radio("View", ["Dashboard", "Assistant", "Search", "Settings"])
    st.divider()

    api_key_override = st.text_input(
        "OpenRouter API key",
        value="",
        type="password",
        placeholder="Uses OPENROUTER_API_KEY from .env",
    )
    api_key = api_key_override or OPENROUTER_API_KEY
    model = st.text_input("OpenRouter model", value=OPENROUTER_MODEL)
    embedding_model = st.text_input("Embedding model", value=EMBEDDING_MODEL)


projects = db.list_projects()
options = project_options(projects)

st.title(APP_NAME)

if page == "Dashboard":
    left, right = st.columns([0.85, 1.15])

    with left:
        st.subheader("Projects")
        with st.form("new_project"):
            name = st.text_input("Project name")
            description = st.text_area("Description", height=90)
            submitted = st.form_submit_button("Create project")
            if submitted:
                if not name.strip():
                    st.error("Project name is required.")
                else:
                    try:
                        db.create_project(name, description)
                        st.success("Project created.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not create project: {exc}")

        if not projects:
            st.info("Create a project, then index files into it.")
        for project in projects:
            files = db.list_files(int(project["id"]))
            with st.expander(f"{project['name']} ({len(files)} files)"):
                st.write(project.get("description") or "No description")
                if files:
                    file_table_data = [
                        {
                            "Select": False,
                            "File Name": f["name"],
                            "Path": f["path"],
                            "Last Modified": f["last_modified"],
                        }
                        for f in files
                    ]
                    edited_data = st.data_editor(
                        file_table_data,
                        hide_index=True,
                        use_container_width=True,
                        disabled=["File Name", "Path", "Last Modified"],
                        key=f"files-editor-{project['id']}",
                    )
                    selected_to_del = [
                        row["Path"] for row in edited_data if row["Select"]
                    ]
                    if st.button(
                        "Delete selected file(s)",
                        key=f"delete-files-{project['id']}",
                        disabled=not selected_to_del,
                    ):
                        get_vector_store().delete_files(int(project["id"]), selected_to_del)
                        db.delete_files(int(project["id"]), selected_to_del)
                        st.success(f"Deleted {len(selected_to_del)} file(s).")
                        st.rerun()
                if st.button("Delete project", key=f"delete-{project['id']}"):
                    get_vector_store().delete_project(int(project["id"]))
                    db.delete_project(int(project["id"]))
                    st.success("Project deleted.")
                    st.rerun()

    with right:
        st.subheader("Index files")
        if not options:
            st.info("Create a project before indexing files.")
        else:
            selected_name = st.selectbox("Project", list(options.keys()))
            selected_project_id = options[selected_name]

            uploaded_files = st.file_uploader(
                "Upload text/code files",
                accept_multiple_files=True,
            )
            if st.button("Index uploaded files", disabled=not uploaded_files):
                docs = []
                for file in uploaded_files or []:
                    content = file.getvalue().decode("utf-8", errors="ignore")
                    docs.append(
                        {
                            "name": file.name,
                            "path": file.name,
                            "content": content,
                            "last_modified": utc_now(),
                        }
                    )
                with st.spinner("Embedding and indexing uploaded files..."):
                    count = index_documents(
                        selected_project_id,
                        docs,
                        embedding_model,
                        api_key=api_key,
                    )
                st.success(f"Indexed {len(docs)} files and {count} chunks.")

            st.divider()
            folder = st.text_input("Or index a local folder path")
            if st.button("Index local folder", disabled=not folder.strip()):
                root = Path(folder).expanduser()
                if not root.exists() or not root.is_dir():
                    st.error("That folder path does not exist.")
                else:
                    paths = iter_text_files(root)
                    docs = []
                    for path in paths:
                        rel_path = str(path.relative_to(root))
                        docs.append(
                            {
                                "name": path.name,
                                "path": rel_path,
                                "content": read_text_file(path),
                                "last_modified": utc_now(),
                            }
                        )
                    with st.spinner("Embedding and indexing folder files..."):
                        count = index_documents(
                            selected_project_id,
                            docs,
                            embedding_model,
                            api_key=api_key,
                        )
                    st.success(f"Indexed {len(docs)} files and {count} chunks.")

elif page == "Assistant":
    if not options:
        st.info("Create and index a project first.")
    else:
        selected_name = st.selectbox("Ask about project", list(options.keys()))
        selected_project_id = options[selected_name]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.chat_input("Ask about your indexed files")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            matches = []
            with st.chat_message("assistant"):
                with st.spinner("Searching and thinking..."):
                    try:
                        embedder = load_embedder(embedding_model, api_key=api_key)
                        query_embedding = embedder.encode([prompt])[0]
                        matches = get_vector_store().search(
                            query_embedding,
                            project_id=selected_project_id,
                            top_k=TOP_K,
                        )
                        history = [
                            item
                            for item in st.session_state.messages[:-1]
                            if item["role"] in {"user", "assistant"}
                        ][-8:]
                        response = answer_with_openrouter(
                            api_key,
                            model,
                            prompt,
                            matches,
                            history,
                        )
                    except Exception as exc:
                        response = f"I could not answer that yet: {exc}"
                st.markdown(response)
                if matches:
                    with st.expander("Sources"):
                        for match in matches:
                            metadata = match["metadata"]
                            st.caption(metadata.get("file_path", "unknown"))
                            st.code(match["document"][:1200])
            st.session_state.messages.append({"role": "assistant", "content": response})

elif page == "Search":
    if not options:
        st.info("Create and index a project first.")
    else:
        selected_name = st.selectbox("Search project", list(options.keys()))
        selected_project_id = options[selected_name]
        query = st.text_input("Semantic search")
        if query:
            embedder = load_embedder(embedding_model, api_key=api_key)
            query_embedding = embedder.encode([query])[0]
            matches = get_vector_store().search(
                query_embedding,
                project_id=selected_project_id,
                top_k=TOP_K,
            )
            for match in matches:
                st.caption(match["metadata"].get("file_path", "unknown"))
                st.code(match["document"][:1600])

else:
    st.subheader("Configuration")
    st.write(
        {
            "database": str(DB_PATH),
            "vector_store": str(VECTOR_DB_DIR),
            "vector_backend": "chroma",
            "embedding_model": embedding_model,
            "chunking": "semantic",
            "semantic_breakpoint_percentile": SEMANTIC_BREAKPOINT_PERCENTILE,
            "openrouter_model": model,
            "openrouter_key_configured": bool(api_key),
        }
    )
