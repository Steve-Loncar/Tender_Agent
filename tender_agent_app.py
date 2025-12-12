import streamlit as st
import json
import time
import requests
from datetime import datetime

# -------------------------------------------------------------------
# Utilities: load core prompt & validate JSON structure
# -------------------------------------------------------------------

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
    st.markdown("### Tender Controls")
    st.markdown(
        "Use this panel to enter tender questions, supply evidence, "
        "and run the core tender-response agent via n8n."
    )
    
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

    st.divider()

    run_button = st.button("Run Tender Agent")

    if run_button:
        if not tender_question.strip():
            st.error("Please enter a tender question before running the agent.")
            st.stop()

        # Build payload for n8n or local simulation
        run_payload = {
            "tender_question": tender_question,
            "authority_name": authority_name,
            "tender_id": tender_id,
            "question_id": question_id,
            "evidence_input": evidence_input,
            "extra_context": extra_context,
            "timestamp_utc": datetime.utcnow().isoformat()
        }

        st.session_state["latest_payload"] = run_payload
        st.success("Tender payload assembled. View the results in the right panel.")

# -------------------------------------------------------------------
# RIGHT PANEL — Output
# -------------------------------------------------------------------
with right_col:
    st.markdown("### Output")

    payload = st.session_state.get("latest_payload")

    if payload:
        st.subheader("Run Payload (Debug View)")
        st.json(payload)

        st.subheader("Simulated Model Output")
        mock_output = {
            "meta": {
                "tender_id": payload["tender_id"],
                "question_id": payload["question_id"],
                "authority_name": payload["authority_name"],
                "agent_version": "tender_core_v1_mock",
                "timestamp_utc": datetime.utcnow().isoformat()
            },
            "question": {
                "original_text": payload["tender_question"],
                "normalised_text": "Normalised form of the tender question...",
                "subquestions": []
            },
            "answer": {
                "high_level_summary": "Mock summary...",
                "sections": [],
                "final_answer_text": "This is a mock response while n8n integration is pending."
            },
            "evidence": [],
            "compliance": {
                "all_subquestions_answered": False,
                "answered_subquestions": [],
                "unanswered_subquestions": [],
                "evidence_coverage_score": 0.0,
                "has_placeholders": False,
                "placeholders_summary": [],
                "hallucination_risk_assessment": "low",
                "risk_flags": [],
                "comments_for_human_reviewer": "This is a mock object used to validate the schema wiring."
            }
        }

        is_valid = is_valid_tender_json(mock_output)
        st.write(f"Schema valid: {is_valid}")
        st.json(mock_output)
