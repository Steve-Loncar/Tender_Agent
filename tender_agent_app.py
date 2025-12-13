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


def _to_list_from_json(v):
    """
    Mirrors the echo app pattern:
    - accept list as-is
    - if string, try json.loads -> list
    - else wrap non-empty string in list
    """
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            return parsed if isinstance(parsed, list) else ([v] if v else [])
        except Exception:
            return [v] if v else []
    return []


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

    # Unwrap n8n response shape: typically [{...}]
    result_obj = None
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
        result_obj = result[0]
    elif isinstance(result, dict):
        result_obj = result

    if not payload and not result_obj:
        st.info("Fill out the tender question and run the agent to see results here.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    # Always keep raw payload accessible, but not dominant
    if payload:
        with st.expander("Payload sent to n8n (raw)"):
            st.json(payload)

    if not result_obj:
        st.info("Payload ready. Click 'Run Tender Agent' to send to n8n.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    # ------------------------------------------------------------
    # Header summary (run_id, status, model, tokens, cost)
    # ------------------------------------------------------------
    store = result_obj.get("store", {}) or {}
    run_id_display = store.get("run_id") or result_obj.get("run_id") or ""
    status = store.get("status") or result_obj.get("status") or ""
    env_mode = store.get("env_mode") or payload.get("env_mode") if payload else ""
    model = store.get("model_name") or result_obj.get("model") or ""
    total_tokens = store.get("total_tokens") or result_obj.get("total_tokens") or ""
    cost_usd = store.get("cost_usd") or result_obj.get("cost_usd") or ""
    ts = store.get("timestamp_utc") or result_obj.get("timestamp") or ""

    top1, top2, top3 = st.columns(3)
    with top1:
        st.metric("Run ID", run_id_display or "—")
        st.caption(f"Status: `{status or '—'}`")
    with top2:
        st.metric("Env", env_mode or "—")
        st.caption(f"Model: `{model or '—'}`")
    with top3:
        st.metric("Tokens", str(total_tokens) if total_tokens != "" else "—")
        try:
            st.caption(f"Cost: ${float(cost_usd):.3f} • {ts}")
        except Exception:
            st.caption(f"Cost: {cost_usd or '—'} • {ts}")

    st.divider()

    # ------------------------------------------------------------
    # Parse the tender JSON (prefer already-parsed; fallback to strings)
    # ------------------------------------------------------------
    tender_json = None
    if isinstance(result_obj.get("llm_output_parsed"), str):
        try:
            tender_json = json.loads(result_obj["llm_output_parsed"])
        except Exception:
            tender_json = None
    if tender_json is None and isinstance(result_obj.get("llm_output_clean"), str):
        try:
            tender_json = json.loads(result_obj["llm_output_clean"])
        except Exception:
            tender_json = None
    if tender_json is None and isinstance(result_obj.get("llm_output_raw"), str):
        try:
            tender_json = json.loads(result_obj["llm_output_raw"])
        except Exception:
            tender_json = None

    # Display run history if available
    if result:
        st.markdown("### Run History (session)")
        run_history = st.session_state.get("run_history", [])
        if run_history:
            df = pd.DataFrame(run_history)
            st.dataframe(df, use_container_width=True, height=220)
        else:
            st.caption("No runs recorded yet in this session.")

    # ------------------------------------------------------------
    # Tabs: Answer / QC / Sources / Debug
    # ------------------------------------------------------------
    tab_answer, tab_qc, tab_sources, tab_debug = st.tabs(
        ["Answer", "QC", "Sources", "Debug"]
    )

    with tab_answer:
        if not isinstance(tender_json, dict):
            st.warning("Could not parse tender JSON from response. Showing raw response in Debug.")
        else:
            # Top summary
            st.subheader("High-level summary")
            st.write(tender_json.get("answer", {}).get("high_level_summary", "—"))

            # Final answer text (human-readable)
            st.subheader("Final answer")
            final_txt = tender_json.get("answer", {}).get("final_answer_text", "")
            if final_txt:
                st.markdown(final_txt)
            else:
                st.info("No final_answer_text field present.")

            # Sections (Q1/Q2/Q3 blocks)
            st.subheader("Answer sections")
            sections = tender_json.get("answer", {}).get("sections", []) or []
            if sections:
                for sec in sections:
                    sid = sec.get("subquestion_id", "")
                    heading = sec.get("heading", "")
                    st.markdown(f"**{sid} — {heading}**")
                    st.write(sec.get("text", ""))
                    ph = sec.get("placeholders_used") or []
                    if ph:
                        with st.expander(f"{sid}: placeholders ({len(ph)})"):
                            for p in ph:
                                st.code(str(p), language="text")
                    st.divider()
            else:
                st.info("No sections found.")

            # Compliance snapshot
            compliance = tender_json.get("compliance", {}) or {}
            st.subheader("Compliance snapshot")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("All subquestions answered", str(compliance.get("all_subquestions_answered", False)))
            with c2:
                st.metric("Evidence coverage", f"{float(compliance.get('evidence_coverage_score', 0.0)):.2f}")
            with c3:
                st.metric("Has placeholders", str(compliance.get("has_placeholders", False)))
            with c4:
                st.metric("Hallucination risk", compliance.get("hallucination_risk_assessment", "n/a"))

    with tab_qc:
        # QC fields come back as JSON strings; mirror echo-app parsing
        qc_issues = _to_list_from_json(result_obj.get("qc_issues_detected"))
        qc_summaries = _to_list_from_json(result_obj.get("qc_issue_summaries"))
        qc_actions = _to_list_from_json(result_obj.get("qc_recommended_actions"))
        qc_append = _to_list_from_json(result_obj.get("qc_suggested_extra_context_append"))

        st.subheader("QC flags")
        if qc_issues:
            st.warning(", ".join([str(x) for x in qc_issues]))
        else:
            st.success("No QC issues detected.")

        st.subheader("QC issue summaries")
        if qc_summaries:
            for s in qc_summaries:
                st.write(f"- {s}")
        else:
            st.caption("No qc_issue_summaries returned.")

        st.subheader("Recommended actions")
        if qc_actions:
            for a in qc_actions:
                st.write(f"- {a}")
        else:
            st.caption("No qc_recommended_actions returned.")

        st.subheader("Suggested extra context to paste into next run")
        if qc_append:
            for a in qc_append:
                st.write(f"- {a}")
        else:
            st.caption("No qc_suggested_extra_context_append returned.")

    with tab_sources:
        st.subheader("Citations")
        citations = result_obj.get("citations") or store.get("citations") or []
        if isinstance(citations, list) and citations:
            for url in citations:
                st.write(f"- {url}")
        else:
            st.caption("No citations list present.")

        st.subheader("Search results (from Perplexity)")
        sr = result_obj.get("search_results") or []
        if isinstance(sr, list) and len(sr) > 0:
            rows = []
            for r in sr:
                if isinstance(r, dict):
                    rows.append(
                        {
                            "title": r.get("title", ""),
                            "date": r.get("date", ""),
                            "last_updated": r.get("last_updated", ""),
                            "url": r.get("url", ""),
                            "source": r.get("source", ""),
                        }
                    )
            if rows:
                st.dataframe(rows, use_container_width=True, height=260)
        else:
            st.caption("No search_results returned.")

        st.subheader("Evidence objects (inside tender JSON)")
        if isinstance(tender_json, dict):
            ev = tender_json.get("evidence", []) or []
            if isinstance(ev, list) and ev:
                ev_rows = []
                for e in ev:
                    if isinstance(e, dict):
                        ev_rows.append(
                            {
                                "evidence_id": e.get("evidence_id", ""),
                                "source_type": e.get("source_type", ""),
                                "source_name": e.get("source_name", ""),
                                "source_reference": e.get("source_reference", ""),
                                "strength_score": e.get("strength_score", ""),
                            }
                        )
                st.dataframe(ev_rows, use_container_width=True, height=220)
            else:
                st.caption("No evidence list in tender JSON.")

    with tab_debug:
        st.subheader("Raw response (n8n)")
        st.json(result_obj)

        if isinstance(tender_json, dict):
            st.subheader("Parsed tender JSON (debug)")
            st.json(tender_json)

        # Preserve the raw strings exactly as received (useful for debugging parser issues)
        with st.expander("llm_output_clean (raw string)"):
            st.text(result_obj.get("llm_output_clean", "")[:20000])
        with st.expander("llm_output_raw (raw string)"):
            st.text(result_obj.get("llm_output_raw", "")[:20000])

    st.markdown("</div>", unsafe_allow_html=True)
