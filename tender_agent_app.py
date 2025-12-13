import streamlit as st
import json
import time
import requests
from datetime import datetime
import uuid
import pandas as pd

# -------------------------------------------------------------------
# Utilities: load core prompt & validate JSON structure
# -------------------------------------------------------------------

# n8n endpoints (MVP: test vs live toggle)
N8N_BASE_URL = "https://fpgconsulting.app.n8n.cloud"
N8N_TEST_PATH = "/webhook-test/tender_agent"
N8N_LIVE_PATH = "/webhook/tender_agent"
WEBHOOK_SECRET = "WIBBLE"


def load_prompt_file(path):
    """Load the tender core prompt from a text file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None


def is_valid_tender_json(obj):
    """
    Very lightweight validation. We only check for required top-level keys.
    Later we can add full field-level validation if needed.
    """
    required = ["meta", "question", "answer", "evidence", "compliance"]
    return isinstance(obj, dict) and all(k in obj for k in required)

# -------------------------------------------------------------------
# Tender Response Agent – Clean Standalone Workspace
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Tender Response Agent (OOH)",
    layout="wide"
)

st.markdown(
    """
    <style>
    /* Two-pane layout helpers */
    .left-pane {
        border-right: 1px solid rgba(255, 255, 255, 0.08);
        padding-right: 0.75rem;
        margin-right: 0.5rem;
    }
    .results-pane {
        background: rgba(245, 247, 250, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 10px;
        padding: 0.75rem 0.9rem;
        box-sizing: border-box;
    }
    .soft-card {
        background: rgba(245, 247, 250, 0.04);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 10px;
        padding: 0.75rem 0.85rem;
    }
    .badge {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        padding: .15rem .45rem;
        border-radius: 8px;
        font-size: .8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("# Tender Response Agent (OOH)")
st.caption(
    "Evidence-based drafting tool for UK public-sector tender questions "
    "relating to Out-of-Home (OOH) advertising services."
)
st.divider()

# -------------------------------------------------------------------
# Layout: 2 columns similar to the A&D app
# -------------------------------------------------------------------
left_col, right_col = st.columns(2, gap="large")

# -------------------------------------------------------------------
# LEFT PANEL — Control Surface (to be expanded in next steps)
# -------------------------------------------------------------------
with left_col:
    st.markdown('<div class="left-pane">', unsafe_allow_html=True)
    st.markdown("### Tender Controls")
    st.markdown(
        "Use this panel to enter tender questions, supply evidence, "
        "and run the core tender-response agent via n8n."
    )

    # ----------------------------------------------------------------
    # Environment switch: test vs live (for development)
    # ----------------------------------------------------------------
    st.subheader("Environment")
    env_choice = st.radio(
        "Select environment",
        options=[
            "Test (webhook-test/tender_agent)",
            "Live (webhook/tender_agent)",
        ],
        index=0,
        help=(
            "Use TEST while developing. This selects the n8n /webhook-test/ "
            "endpoint. Live uses /webhook/ and is for production."
        ),
        horizontal=True,
    )

    if env_choice.startswith("Test"):
        env_mode = "test"
        webhook_url = N8N_BASE_URL + N8N_TEST_PATH
    else:
        env_mode = "live"
        webhook_url = N8N_BASE_URL + N8N_LIVE_PATH

    st.caption(f"Current n8n endpoint: `{webhook_url}`")
    
    st.subheader("Step 1: Provide Tender Question")

    tender_question = st.text_area(
        "Paste the exact tender question below:",
        height=200,
        placeholder="Enter the full text of the tender question here..."
    )

    st.subheader("Step 2: Provide Authority Info (Optional)")
    authority_name = st.text_input("Authority Name", "")
    tender_id = st.text_input("Tender ID (optional)", "")
    question_id = st.text_input("Question ID (optional)", "")

    st.subheader("Step 3: Provide Evidence Inputs")
    st.caption("Paste excerpts or summaries from approved internal documents.")
    evidence_input = st.text_area(
        "Evidence Text",
        height=200,
        placeholder="Paste relevant evidence (SOP extracts, policy lines, KPI snippets)..."
    )

    st.subheader("Step 4: Extra Context (Optional)")
    extra_context = st.text_area(
        "Additional constraints or QC instructions",
        height=150,
        placeholder="Leave empty for now — this will later receive QC critic recommendations."
    )

    st.subheader("Step 5: Core Prompt (auto-loaded)")
    core_prompt = load_prompt_file("prompt_core_tender_json.txt")

    if core_prompt is None:
        st.error("⚠️ Could not find prompt_core_tender_json.txt. Please add it to the app folder.")
    else:
        with st.expander("View core tender prompt"):
            st.code(core_prompt, language="text")

    # --- Global context from UK OOH tender guidance -----------------
    st.subheader("Step 6: Global Tender Context (auto-loaded)")
    global_context = load_prompt_file("prompt_global_context_tenders.txt")

    if global_context is None:
        st.error("⚠️ Could not find prompt_global_context_tenders.txt. Please add it to the app folder.")
        global_context = ""
    else:
        with st.expander("View global context (preview)"):
            preview = global_context[:4000]
            if len(global_context) > 4000:
                preview += "\n\n...[truncated in UI; full context is sent to the agent]..."
            st.code(preview, language="text")

    # ------------------------------------------------------------
    # Step 7: Model & Run Settings
    # ------------------------------------------------------------
    st.subheader("Step 7: Model & Run Settings")

    model_name = st.selectbox(
        "Perplexity model",
        options=[
            "sonar",
            "sonar-small-chat",
            "sonar-pro",
            "sonar-deep-research",
        ],
        index=3,
        help=(
            "Choose the Perplexity model. For complex tender questions with long "
            "context, 'sonar-deep-research' is usually preferred."
        ),
    )

    temperature = st.slider(
        "Temperature (creativity vs determinism)",
        min_value=0.0,
        max_value=0.4,
        value=0.1,
        step=0.05,
        help="Lower = more deterministic and controlled. For tenders, keep this low.",
        key="tender_temperature_slider",
    )

    max_tokens = st.number_input(
        "Max tokens for completion",
        min_value=500,
        max_value=8000,
        value=2000,
        step=100,
        help="Upper bound on tokens for the model's response. Larger values allow longer answers."
    )

    st.divider()

    run_button = st.button("Run Tender Agent")

    if run_button:
        if not tender_question.strip():
            st.error("Please enter a tender question before running the agent.")
            st.stop()

        # Generate a unique run_id for this invocation (mirrors Defence echo app)
        run_id_val = f"run_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex}"
        st.session_state["run_id"] = run_id_val
        st.session_state["last_run_id"] = run_id_val
        st.markdown(
            f'<span class="badge">Run ID</span> <code>{run_id_val}</code>',
            unsafe_allow_html=True
        )

        # ------------------------------------------------------------
        # Build payload for n8n POST
        # ------------------------------------------------------------
        run_payload = {
            "tender_question": tender_question,
            "authority_name": authority_name,
            "tender_id": tender_id,
            "question_id": question_id,
            "evidence_input": evidence_input,
            "extra_context": extra_context,
            "global_context": global_context,
            "qc_critic_prompt": load_prompt_file("prompt_qc_tender.txt"),
            # Environment + endpoint info so n8n / app can route correctly
            "env_mode": env_mode,
            "webhook_url": webhook_url,
            "model_name": model_name,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "timestamp_utc": datetime.utcnow().isoformat(),
            "run_id": run_id_val,
        }

        st.session_state["latest_payload"] = run_payload
        st.info("Sending payload to n8n…")

        # ------------------------------------------------------------
        # POST to n8n webhook
        # ------------------------------------------------------------
        try:
            headers = {
                "Content-Type": "application/json",
                "X-Webhook-Secret": WEBHOOK_SECRET
            }
            
            # Streamlit-side HTTP timeout must exceed n8n end-to-end runtime.
            # Use a (connect, read) tuple to avoid hanging connects.
            if env_mode == "live":
                req_timeout = (10, 120)
            else:
                req_timeout = (10, 60)

            response = requests.post(
                webhook_url,
                headers=headers,
                json=run_payload,
                timeout=req_timeout
            )

            if response.status_code == 200:
                try:
                    st.session_state["n8n_result"] = response.json()
                except Exception:
                    st.session_state["n8n_result"] = {
                        "error": "Received non-JSON response",
                        "raw_text": response.text,
                    }
                st.success("Received response from n8n.")

                # Session run history (echo-style)
                hist = st.session_state.get("run_history", [])
                hist.append(
                    {
                        "run_id": st.session_state.get("last_run_id"),
                        "env": env_mode,
                        "model": model_name,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
                st.session_state["run_history"] = hist
            else:
                st.error(f"n8n returned HTTP {response.status_code}")
                st.session_state["n8n_result"] = {
                    "error": f"HTTP {response.status_code}",
                    "raw_text": response.text,
                }

        except Exception as e:
            st.error(f"Request to n8n failed: {e}")
            st.session_state["n8n_result"] = {
                "error": "Request failed",
                "details": str(e),
            }
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# RIGHT PANEL — Output
# -------------------------------------------------------------------
with right_col:
    st.markdown('<div class="results-pane">', unsafe_allow_html=True)
    st.markdown("### Output")

    payload = st.session_state.get("latest_payload")
    result = st.session_state.get("n8n_result")
    run_id = st.session_state.get("last_run_id") or st.session_state.get("run_id")

    if run_id:
        st.caption(f"Run ID: `{run_id}`")

    if result:
        st.markdown("### Run History (session)")
        run_history = st.session_state.get("run_history", [])
        if run_history:
            df = pd.DataFrame(run_history)
            st.dataframe(df, use_container_width=True, height=220)
        else:
            st.caption("No runs recorded yet in this session.")

        tab_answer, tab_qc, tab_evidence, tab_debug = st.tabs(["Answer", "QC", "Evidence", "Debug"])

        with tab_answer:
            if isinstance(result, dict) and "answer" in result:
                final_text = result.get("answer", {}).get("final_answer_text", "")
                if final_text:
                    st.markdown(final_text)
                else:
                    st.info("No `answer.final_answer_text` found. Showing full answer object.")
                    st.json(result.get("answer"))
            else:
                st.json(result)

        with tab_qc:
            if isinstance(result, dict):
                qc_keys = [
                    "qc_issues_detected",
                    "qc_issue_summaries",
                    "qc_rerun_recommended",
                    "qc_suggested_model",
                    "qc_suggested_temperature",
                    "qc_suggested_max_tokens",
                    "qc_suggested_extra_context_append",
                    "qc_recommended_actions",
                ]
                qc_view = {k: result.get(k) for k in qc_keys if k in result}
                if qc_view:
                    st.json(qc_view)
                else:
                    st.info("No QC fields found in result.")

        with tab_evidence:
            if isinstance(result, dict) and "evidence" in result:
                st.json(result.get("evidence"))
            else:
                st.info("No evidence section found.")

        with tab_debug:
            with st.expander("Show payload sent to n8n", expanded=False):
                if payload:
                    st.code(json.dumps(payload, indent=2), language="json")
                else:
                    st.info("No payload captured yet.")
            with st.expander("Show raw response from n8n", expanded=False):
                st.code(json.dumps(result, indent=2), language="json")
    elif payload:
        st.info("Payload ready. Click 'Run Tender Agent' to send to n8n.")
    else:
        st.info("Fill out the tender question and run the agent to see results here.")

    st.markdown("</div>", unsafe_allow_html=True)
