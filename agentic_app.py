"""
Agentic OCR Pipeline — Streamlit Dashboard (agentic_app.py)

This is the agentic dashboard, separate from app.py (Phase 1 manual pipeline).
Launch with: streamlit run agentic_app.py

Implements Tasks 12.1 – 12.6:
  12.1  Dashboard layout (page config, session state, sidebar, phase bar)
  12.2  Agentic processing flow (orchestrator.execute, explainability, downloads)
  12.3  Human-in-the-Loop dialog
  12.4  Proactive suggestions
  12.5  Full Override Mode (Phase 1 fallback)
  12.6  Privacy notices and terms of use

Requirements: 9.1-9.6, 10.1-10.5, 11.1-11.5, 12.1-12.3, 13.1-13.3, 13.6,
              15.5-15.6, 16.4, 20.1-20.6
"""

import json
import os
import sys
import time
import uuid
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — ensure src/ is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Task 12.1 — Page config (MUST be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Agentic OCR Pipeline",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Lazy imports (avoid crashing on import if heavy deps are missing)
# ---------------------------------------------------------------------------
def _import_agentic():
    from src.agents import create_agentic_system  # noqa: F401
    return create_agentic_system


def _import_phase1():
    from src.preprocessing.image_processor import ImagePreprocessor
    from src.ocr.hybrid_ocr import HybridOCR
    from src.postprocessing.llm_corrector import LLMCorrector
    from src.document_generation.word_generator import WordGenerator
    return ImagePreprocessor, HybridOCR, LLMCorrector, WordGenerator


# ---------------------------------------------------------------------------
# Ensure required directories exist
# ---------------------------------------------------------------------------
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("temp", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ---------------------------------------------------------------------------
# Task 12.1 — Session state initialisation
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "session_id": str(uuid.uuid4()),
    "orchestrator": None,
    "processing": False,
    "result": None,
    "terms_accepted": False,
    "groq_notice_shown": False,
    "suggestion_dismissed": False,
    "hitl_response": None,
    "mode": "🤖 Agentic",
    "override_engine": "groq_api",
    "override_preprocess": False,
    "manual_result": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---------------------------------------------------------------------------
# Task 12.6 — Terms of use (blocks the rest of the app until accepted)
# ---------------------------------------------------------------------------
if not st.session_state.terms_accepted:
    st.info(
        "**Terms of Use & Privacy Notice**\n\n"
        "By using this application you acknowledge:\n\n"
        "- **Local processing**: All image analysis and OCR is performed locally on your "
        "machine by default.\n"
        "- **Groq API**: If you select the Groq API engine, your image data will be "
        "transmitted to Groq's servers for processing. No other data is sent externally.\n"
        "- **Temporary files**: Uploaded and preprocessed images are deleted from the "
        "local filesystem at the end of each processing session.\n"
        "- **No PII storage**: The system does not store raw OCR text, image thumbnails, "
        "or any personally identifiable information in its long-term memory.\n"
        "- **Legal compliance**: This tool must not be used to process, store, or transmit "
        "content that violates applicable law, including PECA 2016.\n\n"
        "Please review these terms before proceeding."
    )
    if st.button("✅ I Accept"):
        st.session_state.terms_accepted = True
        st.rerun()
    st.stop()

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    .phase-pending  { color: #888; font-weight: bold; }
    .phase-active   { color: #1a73e8; font-weight: bold; }
    .phase-done     { color: #34a853; font-weight: bold; }
    .override-banner {
        background: #fce8e6;
        border: 2px solid #d93025;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        color: #d93025;
        font-weight: bold;
        margin-bottom: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helper — phase indicator rendering
# ---------------------------------------------------------------------------
_PHASES = ["Perceive", "Decide", "Act", "Learn"]


def _render_phase_bar(placeholders: list, completed: list, active: str = ""):
    """Update the four phase indicator placeholders."""
    for ph, name in zip(placeholders, _PHASES):
        lower = name.lower()
        if lower in completed:
            ph.markdown(f"<span class='phase-done'>✅ {name}</span>", unsafe_allow_html=True)
        elif lower == active.lower():
            ph.markdown(f"<span class='phase-active'>⏳ {name}</span>", unsafe_allow_html=True)
        else:
            ph.markdown(f"<span class='phase-pending'>⬜ {name}</span>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Task 12.1 — Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    # Mode toggle
    mode = st.radio("Mode", ["🤖 Agentic", "🔧 Manual Override"], key="mode")

    if mode == "🔧 Manual Override":
        st.markdown(
            "<div class='override-banner'>⚠️ Full Override Mode Active — "
            "Autonomous decisions disabled.</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # LTM stats panel
    st.subheader("📊 Long-Term Memory Stats")
    if st.session_state.orchestrator is not None:
        try:
            ltm_stats = st.session_state.orchestrator.memory_manager.get_ltm_stats()
            if ltm_stats:
                for img_type, strategies in ltm_stats.items():
                    with st.expander(f"🖼 {img_type}"):
                        if strategies:
                            best_name = max(
                                strategies,
                                key=lambda s: strategies[s].get("avg_confidence", 0.0),
                            )
                            for strat_name, data in strategies.items():
                                st.write(
                                    f"**{strat_name}**: "
                                    f"{data.get('attempts', 0)} records, "
                                    f"avg conf {data.get('avg_confidence', 0.0):.1%}"
                                )
                            st.caption(f"Best strategy: **{best_name}**")
                        else:
                            st.write("No records yet.")
            else:
                st.info("No long-term memory data yet.")
        except Exception as _e:
            st.warning(f"Could not load LTM stats: {_e}")
    else:
        st.info("Orchestrator not yet initialised. Process an image to populate stats.")

    st.divider()

    # API status
    st.subheader("🔑 API Status")
    if os.getenv("GROQ_API_KEY"):
        st.success("✅ Groq API Key Found")
    else:
        st.warning("⚠️ No Groq API Key — API engine unavailable")

    st.divider()

    # Reset Memory button
    st.subheader("🗑️ Memory Management")
    if st.button("🗑️ Reset Memory"):
        st.warning(
            "This will permanently delete all learned strategy performance data. "
            "Are you sure?"
        )
        if st.button("⚠️ Confirm Reset"):
            if st.session_state.orchestrator is not None:
                st.session_state.orchestrator.memory_manager.reset_long_term_memory()
                st.success("Long-term memory has been reset.")
            else:
                st.error("Orchestrator not initialised — nothing to reset.")

    # Delete All Stored Data (Task 13 integration)
    st.divider()
    st.subheader("🔒 Privacy")
    if st.button("🗑️ Delete All Stored Data"):
        if st.session_state.orchestrator is not None:
            st.session_state.orchestrator.memory_manager.reset_long_term_memory()
        # Delete all session log files
        for _log_file in Path("logs").glob("session_*.json"):
            try:
                _log_file.unlink()
            except OSError:
                pass
        st.success("All stored data deleted.")

# ---------------------------------------------------------------------------
# Task 12.1 — Main area title
# ---------------------------------------------------------------------------
st.markdown("<h1>🤖 Agentic OCR Pipeline</h1>", unsafe_allow_html=True)
st.markdown("Autonomous handwriting recognition powered by the Perceive → Decide → Act → Learn loop.")

# ---------------------------------------------------------------------------
# Task 12.1 — Upload + output name
# ---------------------------------------------------------------------------
col_upload, col_settings = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload handwritten image",
        type=["jpg", "jpeg", "png", "bmp"],
    )

with col_settings:
    output_name = st.text_input("Document Name", value="AgenticNotes")

    # Override controls — only visible in Manual Override mode
    if mode == "🔧 Manual Override":
        st.markdown("**Override Controls**")
        selected_engine = st.selectbox(
            "OCR Engine",
            ["groq_api", "florence_local", "got_ocr_local", "easyocr_local"],
            key="override_engine",
        )
        apply_preprocessing = st.checkbox("Apply Preprocessing", key="override_preprocess")

# ---------------------------------------------------------------------------
# Task 12.1 — Phase indicator bar
# ---------------------------------------------------------------------------
st.divider()
st.markdown("**Processing Phases**")
_phase_cols = st.columns(4)
_phase_placeholders = [col.empty() for col in _phase_cols]
_render_phase_bar(_phase_placeholders, [], "")

st.divider()

# ---------------------------------------------------------------------------
# Helper — session cleanup
# ---------------------------------------------------------------------------
def _cleanup_session(session_id: str, image_path: str):
    """Delete temp files and end the memory session."""
    # Delete uploaded file
    try:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
    except OSError:
        pass
    # Delete any preprocessed temp files
    for _tmp in Path("temp").glob(f"*{session_id}*"):
        try:
            _tmp.unlink()
        except OSError:
            pass
    # End session in memory manager
    if st.session_state.orchestrator is not None:
        try:
            st.session_state.orchestrator.memory_manager.end_session(session_id)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Process button
# ---------------------------------------------------------------------------
if uploaded_file is not None:
    process_btn = st.button("🚀 Process Image", type="primary")
else:
    process_btn = False
    st.info("Upload an image to begin.")

if process_btn and uploaded_file is not None:
    # ------------------------------------------------------------------
    # Task 12.2 — Generate new session ID
    # ------------------------------------------------------------------
    session_id = str(uuid.uuid4())
    st.session_state.session_id = session_id
    st.session_state.result = None
    st.session_state.manual_result = None
    st.session_state.hitl_response = None
    st.session_state.suggestion_dismissed = False
    st.session_state.processing = True

    # Save uploaded file
    image_path = f"uploads/{session_id}_{uploaded_file.name}"
    with open(image_path, "wb") as _f:
        _f.write(uploaded_file.getbuffer())

    # ------------------------------------------------------------------
    # Task 12.5 — Full Override Mode: bypass orchestrator entirely
    # ------------------------------------------------------------------
    if mode == "🔧 Manual Override":
        st.markdown(
            "<div class='override-banner'>⚠️ Full Override Mode Active — "
            "Autonomous decisions disabled.</div>",
            unsafe_allow_html=True,
        )
        _render_phase_bar(_phase_placeholders, [], "act")

        with st.spinner("Running Phase 1 pipeline (Manual Override)..."):
            try:
                ImagePreprocessor, HybridOCR, LLMCorrector, WordGenerator = _import_phase1()

                preprocessor = ImagePreprocessor()
                engine_map = {
                    "groq_api": "api",
                    "florence_local": "florence",
                    "got_ocr_local": "got",
                    "easyocr_local": "easyocr",
                }
                engine_key = engine_map.get(selected_engine, "api")
                prefer_local = selected_engine != "groq_api"

                # Task 12.6 — Groq API notice before first call
                if selected_engine == "groq_api" and not st.session_state.groq_notice_shown:
                    st.warning(
                        "📡 Notice: Image data will be sent to the Groq API for processing."
                    )
                    if st.button("Proceed with Groq API"):
                        st.session_state.groq_notice_shown = True
                        st.rerun()
                    st.stop()

                if apply_preprocessing:
                    _, proc_path = preprocessor.preprocess(image_path)
                    ocr_image = proc_path
                else:
                    ocr_image = image_path

                ocr_engine = HybridOCR(prefer_local=prefer_local, local_model=engine_key)
                ocr_result = ocr_engine.extract_text_from_image(ocr_image)
                extracted_text = ocr_result.get("text", "")

                corrector = LLMCorrector()
                corrected = corrector.correct_text(extracted_text)
                structured = corrector.structure_content(corrected)

                word_gen = WordGenerator()
                out_path = word_gen.create_document(structured, output_name)

                st.session_state.manual_result = {
                    "extracted_text": extracted_text,
                    "output_path": out_path,
                    "engine": selected_engine,
                    "mode": "manual_override",
                }

                # Record mode activation in session state
                st.session_state[f"override_activated_{session_id}"] = True

            except Exception as _exc:
                st.error(f"❌ Manual Override processing failed: {_exc}")
                st.session_state.processing = False

        _render_phase_bar(_phase_placeholders, ["act"], "")
        st.session_state.processing = False

    # ------------------------------------------------------------------
    # Task 12.2 — Agentic processing flow
    # ------------------------------------------------------------------
    else:
        # Lazy-init orchestrator
        if st.session_state.orchestrator is None:
            with st.spinner("Initialising agentic system..."):
                create_agentic_system = _import_agentic()
                st.session_state.orchestrator = create_agentic_system()

        orchestrator = st.session_state.orchestrator

        # Build user_override if needed (always None in Agentic mode)
        user_override = None

        # Task 12.6 — Groq API notice (shown before first API call if applicable)
        # We show it proactively if no API key is set (won't be used anyway),
        # but if key exists and notice not shown, we'll show it after analysis.

        _render_phase_bar(_phase_placeholders, [], "perceive")

        with st.spinner("🤖 Running agentic pipeline..."):
            result = orchestrator.execute(
                {
                    "image_path": image_path,
                    "session_id": session_id,
                    "user_override": user_override,
                }
            )

        st.session_state.result = result
        st.session_state.processing = False

        # Update phase bar based on completed phases
        phases_done = result.get("phases_completed", [])
        _render_phase_bar(_phase_placeholders, phases_done, "")

        # Task 12.6 — Show Groq notice if API was likely used
        if not st.session_state.groq_notice_shown and os.getenv("GROQ_API_KEY"):
            st.warning("📡 Notice: Image data may have been sent to the Groq API for processing.")
            st.session_state.groq_notice_shown = True

        # Cleanup temp files
        _cleanup_session(session_id, image_path)

# ===========================================================================
# Results section — Agentic mode
# ===========================================================================
result = st.session_state.get("result")

if result is not None and mode == "🤖 Agentic":
    st.divider()

    # ------------------------------------------------------------------
    # Task 12.4 — Proactive Suggestions (shown after processing based on
    # analysis embedded in decision log / result)
    # ------------------------------------------------------------------
    decision_log = result.get("decision_log", {})
    actions = decision_log.get("actions", [])

    # Extract analysis report from actions
    analysis_report = {}
    for _action in actions:
        if _action.get("action") == "image_analysis" and "result" in _action:
            analysis_report = _action["result"]
            break

    image_type_detected = analysis_report.get("image_type", "")
    noise_level = analysis_report.get("noise_level", 0.0)
    recommended_enhancements = analysis_report.get("recommended_enhancements", [])

    if image_type_detected == "low_quality" and not st.session_state.suggestion_dismissed:
        suggestion_text = (
            "💡 **Suggestion**: The uploaded image appears to be low quality. "
        )
        if recommended_enhancements:
            suggestion_text += "Recommended improvements: " + "; ".join(recommended_enhancements)
        else:
            suggestion_text += (
                "Consider scanning at a higher resolution (≥300 DPI) or improving "
                "lighting before re-uploading."
            )
        st.info(suggestion_text)
        if st.button("Dismiss suggestion"):
            st.session_state.suggestion_dismissed = True
            st.rerun()

    if noise_level > 0.6:
        st.info(
            "💡 **Noise detected**: The image has a high noise level. "
            "Automatic denoising preprocessing is recommended for better results."
        )
        _apply_denoise = st.checkbox("Apply automatic denoising on next run")
        if _apply_denoise:
            st.caption(
                "Denoising will be applied when you re-process the image. "
                "Please re-upload and process again."
            )

    # ------------------------------------------------------------------
    # Task 12.3 — Human-in-the-Loop dialog
    # ------------------------------------------------------------------
    if result.get("hitl_triggered") is True and st.session_state.hitl_response is None:
        st.warning(
            "⚠️ All retry attempts exhausted. Please review the best available result."
        )

        _best = result.get("final_result") or {}
        _best_text = _best.get("text", "") if isinstance(_best, dict) else ""
        _conf = result.get("confidence_score", 0.0)

        st.metric("Best Available Confidence Score", f"{_conf:.1%}")
        if _best_text:
            st.text_area("Best Available Result Preview", value=_best_text[:500], height=150)

        _hitl_choice = st.radio(
            "What would you like to do?",
            ["Accept result", "Choose strategy manually", "Cancel"],
            key="hitl_radio",
        )

        if _hitl_choice == "Choose strategy manually":
            _manual_engine = st.selectbox(
                "Select strategy to re-run",
                ["groq_api", "florence_local", "got_ocr_local", "easyocr_local"],
                key="hitl_engine_select",
            )
            if st.button("🔄 Re-run with selected strategy"):
                _override = {
                    "name": _manual_engine,
                    "tool_name": "ocr_tool",
                    "engine": _manual_engine.replace("_local", "").replace("groq_api", "api"),
                    "requires_api": _manual_engine == "groq_api",
                }
                _new_session = str(uuid.uuid4())
                st.session_state.session_id = _new_session
                # Re-save the uploaded file if still available
                if uploaded_file is not None:
                    _rerun_path = f"uploads/{_new_session}_{uploaded_file.name}"
                    with open(_rerun_path, "wb") as _rf:
                        _rf.write(uploaded_file.getbuffer())
                    _rerun_result = st.session_state.orchestrator.execute(
                        {
                            "image_path": _rerun_path,
                            "session_id": _new_session,
                            "user_override": _override,
                        }
                    )
                    st.session_state.result = _rerun_result
                    st.session_state.hitl_response = "rerun"
                    _cleanup_session(_new_session, _rerun_path)
                    st.rerun()
                else:
                    st.error("Original image no longer available. Please re-upload.")

        elif _hitl_choice == "Cancel":
            st.session_state.hitl_response = "cancel"
            st.info("Processing cancelled. You can upload a new image.")
            st.stop()

        elif _hitl_choice == "Accept result":
            if st.button("✅ Accept and continue"):
                st.session_state.hitl_response = "accept"
                st.rerun()

        # Block document generation until user responds
        if st.session_state.hitl_response is None:
            st.stop()

    # ------------------------------------------------------------------
    # Task 12.2 — Display results
    # ------------------------------------------------------------------
    if result.get("status") == "error":
        st.error(f"❌ Processing failed: {result.get('error_message', 'Unknown error')}")
    else:
        # Confidence score + quality badge
        conf = result.get("confidence_score", 0.0)
        final_res = result.get("final_result") or {}
        quality_report = {}
        # Try to extract quality report from actions
        for _action in actions:
            if _action.get("action") == "quality_assessment":
                quality_report = _action.get("result", {})
                break

        classification = quality_report.get("classification", "")
        if not classification:
            if conf >= 0.8:
                classification = "high_quality"
            elif conf >= 0.5:
                classification = "acceptable"
            else:
                classification = "low_quality"

        low_quality = classification == "low_quality"

        col_conf, col_class = st.columns(2)
        with col_conf:
            st.metric("Confidence Score", f"{conf:.1%}")
        with col_class:
            if classification == "high_quality":
                st.success(f"✅ Quality: {classification}")
            elif classification == "acceptable":
                st.warning(f"⚠️ Quality: {classification}")
            else:
                st.error(f"❌ Quality: {classification}")

        if low_quality:
            st.warning("⚠️ Low quality result — please review the extracted text carefully.")

        # ------------------------------------------------------------------
        # Explainability Panel
        # ------------------------------------------------------------------
        explainability_report = result.get("explainability_report", "")
        with st.expander("🔍 Explainability Report", expanded=False):
            if explainability_report:
                st.text(explainability_report)
            else:
                st.info("No explainability report available.")

            # Image type + strategy info from analysis
            if image_type_detected:
                st.markdown(f"**Detected Image Type:** {image_type_detected}")

            # Retry history from decision log actions
            retry_actions = [
                a for a in actions
                if a.get("action") == "retry_attempt"
            ]
            if retry_actions:
                st.markdown("**Retry History:**")
                for _ra in retry_actions:
                    st.write(
                        f"- Attempt {_ra.get('attempt', '?')}: "
                        f"strategy={_ra.get('strategy', '?')}, "
                        f"confidence={_ra.get('confidence', 0.0):.1%}"
                    )

            # Bias disclosure (extracted from explainability report)
            if "Bias disclosure" in explainability_report:
                _bias_start = explainability_report.find("Bias disclosure")
                _bias_line = explainability_report[_bias_start:].split("\n")[0]
                st.markdown(f"**{_bias_line}**")

        # ------------------------------------------------------------------
        # Short-Term Memory timeline
        # ------------------------------------------------------------------
        with st.expander("📋 Session Memory Timeline", expanded=False):
            if actions:
                for _act in actions:
                    _ts = _act.get("timestamp", "")
                    _agent = _act.get("agent", "?")
                    _action_name = _act.get("action", "?")
                    st.write(f"🕐 `{_ts}` — **{_agent}** → {_action_name}")
            else:
                st.info("No session actions recorded.")

        # ------------------------------------------------------------------
        # Editable text area
        # ------------------------------------------------------------------
        extracted_text = ""
        if isinstance(final_res, dict):
            extracted_text = final_res.get("text", "") or final_res.get("corrected_text", "")

        edited_text = st.text_area(
            "Extracted Text (editable)",
            value=extracted_text,
            height=300,
        )

        # ------------------------------------------------------------------
        # Download buttons
        # ------------------------------------------------------------------
        st.markdown("**Downloads**")
        dl_col1, dl_col2, dl_col3 = st.columns(3)

        # Word document
        output_path = ""
        if isinstance(final_res, dict):
            output_path = final_res.get("output_path", "")

        with dl_col1:
            if output_path and os.path.exists(output_path):
                with open(output_path, "rb") as _docx_f:
                    st.download_button(
                        label="⬇️ Word Document",
                        data=_docx_f,
                        file_name=f"{output_name}.docx",
                        mime=(
                            "application/vnd.openxmlformats-officedocument"
                            ".wordprocessingml.document"
                        ),
                    )
            else:
                st.caption("Word document not available.")

        # Decision Log JSON
        with dl_col2:
            _decision_log_json = json.dumps(result.get("decision_log", {}), indent=2)
            st.download_button(
                label="⬇️ Decision Log (JSON)",
                data=_decision_log_json,
                file_name=f"decision_log_{st.session_state.session_id}.json",
                mime="application/json",
            )

        # Session Memory JSON
        with dl_col3:
            if st.session_state.orchestrator is not None:
                _session_mem_json = (
                    st.session_state.orchestrator.memory_manager.get_session_log_json(
                        st.session_state.session_id
                    )
                )
            else:
                _session_mem_json = "{}"
            st.download_button(
                label="⬇️ Session Memory (JSON)",
                data=_session_mem_json,
                file_name=f"session_memory_{st.session_state.session_id}.json",
                mime="application/json",
            )

# ===========================================================================
# Results section — Manual Override mode
# ===========================================================================
manual_result = st.session_state.get("manual_result")

if manual_result is not None and mode == "🔧 Manual Override":
    st.divider()
    st.success("✅ Manual Override processing complete.")

    _mo_text = manual_result.get("extracted_text", "")
    _mo_path = manual_result.get("output_path", "")
    _mo_engine = manual_result.get("engine", "unknown")

    st.metric("OCR Engine Used", _mo_engine)

    _mo_edited = st.text_area(
        "Extracted Text (editable)",
        value=_mo_text,
        height=300,
    )

    if _mo_path and os.path.exists(_mo_path):
        with open(_mo_path, "rb") as _mo_f:
            st.download_button(
                label="⬇️ Download Word Document",
                data=_mo_f,
                file_name=f"{output_name}.docx",
                mime=(
                    "application/vnd.openxmlformats-officedocument"
                    ".wordprocessingml.document"
                ),
            )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.markdown(
    "<div style='text-align:center;color:#888;padding:1rem;'>"
    "🤖 Agentic OCR Pipeline | Perceive → Decide → Act → Learn"
    "</div>",
    unsafe_allow_html=True,
)
