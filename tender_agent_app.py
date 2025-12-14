import streamlit as st
import json
import time
import streamlit.components.v1 as components
import requests
import re
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

# Status agent endpoints (poll-by-run_id; mirrors Echo/Defence pattern)
N8N_STATUS_TEST_PATH = "/webhook-test/tender_status"
N8N_STATUS_LIVE_PATH = "/webhook/tender_status"


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

def _dedupe_keep_order(items):
    seen = set()
    out = []
    for it in items or []:
        key = str(it).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out

def _score_citation_url(url: str) -> int:
    """
    Heuristic ranking so the *useful* OOH / market-size sources float to the top.
    Keeps everything (nothing is discarded), just re-ordered.
    """
    if not url:
        return 0
    u = url.lower()
    score = 0
    # Strongly relevant for UK OOH market sizing
    if "outsmart" in u:
        score += 5
    if "warc" in u or "advertisingassociation" in u or "aa/" in u:
        score += 5
    # Often useful for spend / market context
    if "ipa.co.uk" in u or "nielsen" in u or "statista" in u:
        score += 2
    # Procurement-only / tender mechanics (can be useful, but not for market size)
    if "gov.uk" in u and ("procurement" in u or "tender" in u or "contracts" in u):
        score -= 1
    if "tendersdirect" in u or "tenderconsultants" in u:
        score -= 2
    return score

def _extract_urls(text: str) -> list:
    if not text:
        return []
    return re.findall(r"https?://[^\s)>\"]+", text)


def _copy_to_clipboard_button(text: str, label: str, key: str):
    """
    Streamlit doesn't have a native 'copy' button for arbitrary text.
    This injects a tiny HTML/JS snippet to copy the provided text.
    """
    safe = (text or "").replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    html = f"""
    <div style="margin: 0.25rem 0 0.75rem 0;">
      <button
        id="{key}"
        style="
          padding: 0.35rem 0.6rem;
          border-radius: 0.5rem;
          border: 1px solid rgba(255,255,255,0.2);
          background: rgba(255,255,255,0.06);
          color: inherit;
          cursor: pointer;
        "
      >
        {label}
      </button>
      <span id="{key}_status" style="margin-left: 0.6rem; opacity: 0.8;"></span>
    </div>
    <script>
      const btn = document.getElementById("{key}");
      const status = document.getElementById("{key}_status");
      btn.addEventListener("click", async () => {{
        try {{
          await navigator.clipboard.writeText(`{safe}`);
          status.textContent = "Copied ✓";
          setTimeout(() => status.textContent = "", 1500);
        }} catch (e) {{
          status.textContent = "Copy failed";
          setTimeout(() => status.textContent = "", 2000);
        }}
      }});
    </script>
    """
    components.html(html, height=60)


def display_formatted_text(text: str, max_sentence_length: int = 200):
    """
    Display large text blocks in a readable format using Streamlit components.
    Breaks long paragraphs into bullet points with proper spacing.
    """
    if not text or len(text) < 300:  # Short text doesn't need formatting
        st.markdown(text)
        return
    
    # Split into paragraphs first
    paragraphs = text.split('\n\n')
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # If paragraph is already bulleted or numbered, keep as is
        if para.startswith(('• ', '- ', '* ', '1. ', '2. ', '3. ')) or para.count('\n• ') > 0:
            st.markdown(para)
            continue
            
        # For long paragraphs, split into sentences and display as bullets
        if len(para) > max_sentence_length:
            # Split on sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', para)
            if len(sentences) > 2:  # Only bulletize if multiple sentences
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence:
                        st.write(f"• {sentence}")
                st.write("")  # Add spacing after bullet section
            else:
                st.markdown(para)
        else:
            st.markdown(para)
        
        st.write("")  # Add spacing between paragraphs


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
        status_url = N8N_BASE_URL + N8N_STATUS_TEST_PATH
    else:
        env_mode = "live"
        webhook_url = N8N_BASE_URL + N8N_LIVE_PATH
        status_url = N8N_BASE_URL + N8N_STATUS_LIVE_PATH

    st.caption(f"Current n8n endpoint: `{webhook_url}`")
    st.caption(f"Current status endpoint: `{status_url}`")
    
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
        index=2,
        help=(
            "Choose the Perplexity model. For complex tender questions with long "
            "context, 'sonar-deep-research' is usually preferred."
        ),
    )

    # Model-aware default completion budget (kept editable by the user)
    default_max_tokens_by_model = {
        "sonar": 2500,
        "sonar-small-chat": 2000,
        "sonar-pro": 3500,
        "sonar-deep-research": 4500,
    }
    default_max_tokens = default_max_tokens_by_model.get(model_name, 3000)

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
        value=int(default_max_tokens),
        step=100,
        help="Upper bound on tokens for the model's response. Larger values allow longer answers."
    )

    st.divider()

    # Echo/Defence style: status poll + run controls side-by-side
    run_col1, run_col2 = st.columns([1, 1])

    with run_col1:
        st.caption("Poll a past run (manual Run ID)")
        manual_poll_run_id = st.text_input(
            "Run ID to poll (optional)",
            value=st.session_state.get("manual_poll_run_id", ""),
            placeholder="e.g. run_20251213T215711_4e4b8548be5a41eab25951e9d5fa19c9",
            key="manual_poll_run_id_input",
        )
        if manual_poll_run_id is not None:
            st.session_state["manual_poll_run_id"] = manual_poll_run_id.strip()

        if st.button("Check for completed result"):
            # Prefer manually pasted Run ID; fallback to latest run in this session
            last_run_id = (
                (st.session_state.get("manual_poll_run_id") or "").strip()
                or st.session_state.get("last_run_id")
                or st.session_state.get("run_id")
            )
            if not last_run_id:
                st.warning("No Run ID available. Run the agent first (or paste a Run ID into session).")
            else:
                st.info(f"Checking status for Run ID: `{last_run_id}`")
                headers = {"Content-Type": "application/json"}
                if WEBHOOK_SECRET:
                    headers["X-Webhook-Secret"] = WEBHOOK_SECRET
                try:
                    status_resp = requests.post(
                        status_url,
                        json={"run_id": last_run_id},
                        headers=headers,
                        timeout=30,
                    )
                    st.write("Status check HTTP code:", status_resp.status_code)
                    try:
                        sj = status_resp.json()
                    except Exception:
                        st.error("Status endpoint did not return JSON.")
                        st.text(status_resp.text[:1200] if status_resp.text else "")
                    else:
                        # Treat explicit pending as "still running"; anything else as completed payload
                        status_val = ""
                        if isinstance(sj, dict) and "status" in sj:
                            status_val = str(sj.get("status", "")).lower()
                        if isinstance(sj, dict) and status_val == "pending":
                            st.info("Still running inside n8n/Perplexity. Try again shortly.")
                        else:
                            # Store as the main result so the existing right-pane rendering works
                            st.session_state["n8n_result"] = sj
                            st.success("Retrieved completed result from tender_status.")
                except Exception as e:
                    st.error(f"Status check failed: {e}")

    with run_col2:
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
            "prompt_text": core_prompt,
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
            "status_url": status_url,
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
            else:
                st.error(f"n8n returned HTTP {response.status_code}")
                st.session_state["n8n_result"] = {
                    "error": f"HTTP {response.status_code}",
                    "raw_text": response.text,
                }

        except (requests.exceptions.ReadTimeout, requests.exceptions.Timeout):
            # Key behaviour for sonar-deep-research: Perplexity/n8n can continue after Streamlit times out.
            st.warning(
                "Streamlit timed out waiting for n8n, but the run may still be executing. "
                "Use **Check for completed result** to poll by Run ID."
            )
            st.session_state["n8n_result"] = {
                "status": "pending",
                "run_id": run_id_val,
                "message": "Timed out waiting for n8n response; poll tender_status for completion."
            }

        except Exception as e:
            st.error(f"Request to n8n failed: {e}")
            st.session_state["n8n_result"] = {
                "error": "Request failed",
                "details": str(e),
            }

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
            summary_text = tender_json.get("answer", {}).get("high_level_summary", "—")
            if summary_text and summary_text != "—":
                display_formatted_text(summary_text, max_sentence_length=150)
            else:
                st.write(summary_text)

            # Final answer text (human-readable)
            st.subheader("Final answer")
            final_txt = tender_json.get("answer", {}).get("final_answer_text", "")
            if final_txt:
                display_formatted_text(final_txt)
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
                    
                    section_text = sec.get("text", "")
                    if section_text:
                        display_formatted_text(section_text)
                    
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

        # New QC fields (non-blocking reviewer value)
        qc_strengths = _to_list_from_json(result_obj.get("qc_strengths_summary"))
        qc_optional = _to_list_from_json(result_obj.get("qc_optional_improvements"))

        # Rerun recommendation + tuning
        qc_rerun = bool(result_obj.get("qc_rerun_recommended", False))
        qc_model = result_obj.get("qc_suggested_model", None)
        qc_temp = result_obj.get("qc_suggested_temperature", None)
        qc_tok = result_obj.get("qc_suggested_max_tokens", None)

        st.subheader("Rerun recommendation (at a glance)")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Rerun recommended", "YES" if qc_rerun else "NO")
        with c2:
            st.metric("Suggested model", str(qc_model) if qc_model is not None else "n/a")
        with c3:
            st.metric("Suggested temp", str(qc_temp) if qc_temp is not None else "n/a")
        with c4:
            st.metric("Suggested max_tokens", str(qc_tok) if qc_tok is not None else "n/a")

        if qc_rerun:
            st.warning(
                "QC recommends a rerun. Apply the suggested model/temperature/max_tokens and paste the extra context below into the next run."
            )
        else:
            st.success(
                "QC does not require a rerun. Review optional improvements below if you want to further polish."
            )

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

        st.subheader("Strengths (what is working well)")
        if qc_strengths:
            for s in qc_strengths:
                st.write(f"- {s}")
        else:
            st.caption("No qc_strengths_summary returned.")

        st.subheader("Optional improvements (non-blocking)")
        if qc_optional:
            for s in qc_optional:
                st.write(f"- {s}")
        else:
            st.caption("No qc_optional_improvements returned.")

        st.subheader("Recommended actions")
        if qc_actions:
            for a in qc_actions:
                st.write(f"- {a}")
        else:
            st.caption("No qc_recommended_actions returned.")

        st.subheader("Suggested extra context to paste into next run")
        extra_context_text = ""
        if qc_append:
            # Make this easy to copy/paste as a single block
            extra_context_text = "\n".join([str(x) for x in qc_append if str(x).strip()])

        if extra_context_text.strip():
            st.text_area(
                "Extra context (copy/paste into next run)",
                value=extra_context_text,
                height=220,
                help="This is generated by QC to be appended to the next run's extra_context."
            )
            _copy_to_clipboard_button(
                extra_context_text,
                label="Copy extra context",
                key="copy_qc_extra_context"
            )
        else:
            st.caption("No qc_suggested_extra_context_append returned.")

    with tab_sources:
        # ------------------------------------------------------------
        # Evidence-first display (like the defence/echo app style):
        # 1) show structured evidence + what it supports
        # 2) show top citations ranked (OOH / AA/WARC / Outsmart first)
        # 3) keep raw search_results in an expander (for audit/debug)
        # ------------------------------------------------------------

        # -------- Evidence objects (primary, inside tender JSON) -------
        st.subheader("Primary evidence used in the answer")
        ev = []
        if isinstance(tender_json, dict):
            ev = tender_json.get("evidence", []) or []

        # Sort evidence: display_priority asc, strength_score desc (echo-style clarity)
        if isinstance(ev, list) and ev:
            def _ev_sort_key(e):
                if not isinstance(e, dict):
                    return (9999, 0)
                dp = e.get("display_priority")
                try:
                    dpv = int(dp) if dp is not None else 9999
                except Exception:
                    dpv = 9999
                try:
                    ss = float(e.get("strength_score", 0.0))
                except Exception:
                    ss = 0.0
                return (dpv, -ss)
            ev = sorted(ev, key=_ev_sort_key)

        # Map evidence_id -> where it was used (from answer.sections[].evidence_ids)
        used_map = {}
        if isinstance(tender_json, dict):
            sections = tender_json.get("answer", {}).get("sections", []) or []
            for sec in sections:
                sid = sec.get("subquestion_id", "")
                for eid in sec.get("evidence_ids") or []:
                    used_map.setdefault(str(eid), set()).add(str(sid))

        if isinstance(ev, list) and ev:
            for e in ev:
                if not isinstance(e, dict):
                    continue
                eid = str(e.get("evidence_id", "")).strip() or "E?"
                title = (e.get("source_name") or "").strip() or "Evidence item"
                st.markdown(f"**{eid} — {title}**")

                meta_bits = []
                if e.get("source_type"):
                    meta_bits.append(str(e.get("source_type")))
                if e.get("internal_or_external"):
                    meta_bits.append(str(e.get("internal_or_external")))
                if e.get("strength_score") != "" and e.get("strength_score") is not None:
                    meta_bits.append(f"strength={e.get('strength_score')}")
                if meta_bits:
                    st.caption(" • ".join(meta_bits))

                # What does this support?
                supports = []
                # Prefer explicit relevant_subquestions if present; fallback to where it was used
                rsq = e.get("relevant_subquestions") or []
                if isinstance(rsq, list) and rsq:
                    supports = [str(x) for x in rsq]
                elif eid in used_map:
                    supports = sorted(list(used_map[eid]))
                if supports:
                    st.markdown(f"<small><b>Supports:</b> {', '.join(supports)}</small>", unsafe_allow_html=True)

                # Show quote/paraphrase for traceability (short)
                quote = (e.get("quote") or "").strip()
                paraphrase = (e.get("paraphrase") or "").strip()
                if quote:
                    st.caption(f"Quote: {quote[:450]}{'…' if len(quote) > 450 else ''}")
                elif paraphrase:
                    st.caption(f"Note: {paraphrase[:450]}{'…' if len(paraphrase) > 450 else ''}")

                # Show explicit claim mapping if provided (new schema field already in core prompt)
                supports_claims = e.get("supports_claims") or []
                if isinstance(supports_claims, list) and supports_claims:
                    with st.expander(f"{eid}: what this evidence supports"):
                        for sc in supports_claims:
                            if not isinstance(sc, dict):
                                continue
                            claim = (sc.get("claim") or "").strip()
                            locs = sc.get("answer_locations") or []
                            locs_txt = ", ".join([str(x) for x in locs]) if isinstance(locs, list) else str(locs)
                            if claim:
                                st.markdown(f"- **Claim:** {claim}")
                                if locs_txt:
                                    st.caption(f"Appears in: {locs_txt}")

                # Link out if evidence has a concrete URL in source_reference
                src_ref = (e.get("source_reference") or "").strip()
                if src_ref.startswith("http"):
                    st.markdown(f"- Source: [{src_ref}]({src_ref})")
                # Prefer dedicated source_url if present
                src_url = (e.get("source_url") or "").strip() if isinstance(e.get("source_url"), str) else ""
                if src_url.startswith("http"):
                    st.markdown(f"- Source URL: [{src_url}]({src_url})")
                st.divider()
        else:
            st.caption("No structured evidence items returned in tender JSON.")

        # -------- Ranked citations (deduped) ---------------------------
        st.subheader("Top sources (ranked)")

        # Prefer URLs from structured evidence (these are actually used in-answer)
        evidence_urls = []
        if isinstance(ev, list) and ev:
            for e in ev:
                if isinstance(e, dict):
                    u = e.get("source_url")
                    if isinstance(u, str) and u.strip():
                        evidence_urls.append(u.strip())
                    sr = e.get("source_reference")
                    if isinstance(sr, str) and sr.strip().startswith("http"):
                        evidence_urls.append(sr.strip())
        evidence_urls = _dedupe_keep_order(evidence_urls)

        citations = result_obj.get("citations") or store.get("citations") or []
        citations = [c for c in citations if isinstance(c, str)]
        citations = _dedupe_keep_order([c.strip() for c in citations if c.strip()])

        # Merge (evidence URLs first, then remaining citations), then rank
        merged_sources = _dedupe_keep_order((evidence_urls or []) + (citations or []))

        if merged_sources:
            ranked = sorted(merged_sources, key=lambda u: _score_citation_url(u), reverse=True)
            top = ranked[:6]
            rest = ranked[6:]

            for url in top:
                st.markdown(f"- [{url}]({url})")

            if rest:
                with st.expander(f"More sources ({len(rest)})"):
                    for url in rest:
                        st.markdown(f"- [{url}]({url})")
        else:
            st.caption("No citations list present.")

        # -------- Search results (audit/debug only) --------------------
        sr = result_obj.get("search_results") or []
        if isinstance(sr, list) and len(sr) > 0:
            with st.expander("Search results (from Perplexity)"):
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
                    st.caption("No usable search_results rows.")
        else:
            st.caption("No search_results returned.")

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
