# Implementation Plan: Agentic OCR Pipeline

## Overview

Transform the Phase 1 static OCR pipeline into a Purely Agentic System by implementing the Perceive → Decide → Act → Learn loop. All Phase 1 components (`src/preprocessing/`, `src/ocr/`, `src/postprocessing/`, `src/document_generation/`) remain unchanged. New code lives in `src/agents/`, `tests/`, `memory/`, `logs/`, and `agentic_app.py`.

All implementation uses **Python**.

---

## Tasks

- [x] 1. Set up project foundation — directories, base classes, and type definitions
  - Create `src/agents/__init__.py`, `src/agents/tools/__init__.py`
  - Create `memory/` and `logs/` directories (add `.gitkeep` files)
  - Create `tests/__init__.py`, `tests/unit/__init__.py`, `tests/property/__init__.py`, `tests/integration/__init__.py`
  - Create `src/agents/base_agent.py` with the `BaseAgent` abstract class exactly as specified in the design: `__init__(name, memory_manager)`, abstract `execute(inputs) -> dict`, `log_decision()`, and `handle_error()` methods
  - Create `src/agents/tools/base_tool.py` with the `AgentTool` abstract class: `invoke(inputs) -> dict` (catches all exceptions, returns `status/output/execution_time_seconds`), and abstract `_execute(inputs)` method
  - Add data model type definitions (TypedDicts: `ImageAnalysisReport`, `Strategy`, `OCRResult`, `QualityReport`, `DecisionLog`, `ShortTermMemory`, `LongTermMemory`) as a module `src/agents/models.py`
  - _Requirements: 19.1, 19.2, 8.2_

- [ ] 2. Implement Memory Manager
  - [x] 2.1 Create `src/agents/memory_manager.py` implementing `MemoryManager` as specified in the design
    - `__init__`: creates `memory/` and `logs/` dirs, loads LTM from `memory/long_term_memory.json`
    - `start_session(session_id, image_path)`: initialises STM dict for the session
    - `end_session(session_id)`: writes `logs/session_{id}.json`, clears STM entry
    - `record_action(action)`, `record_decision(decision)`, `record_tried_strategy(session_id, strategy_name)`, `was_strategy_tried(session_id, strategy_name) -> bool`
    - `get_session_log(session_id) -> dict`, `get_session_log_json(session_id) -> str`
    - `get_long_term_memory() -> dict`, `update_long_term_memory(image_type, strategy, confidence_score, success)` with rolling window capped at 50
    - `get_ltm_stats() -> dict`, `reset_long_term_memory()`
    - `_load_long_term_memory()`: handles missing file and `json.JSONDecodeError` gracefully (initialises fresh, logs warning)
    - `_save_long_term_memory()`: writes to `memory/long_term_memory.json`
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 13.4_

  - [ ]* 2.2 Write property test for LTM success rates always valid (Property 6)
    - **Property 6: Long-Term Memory Success Rates Always Valid**
    - File: `tests/property/test_memory_properties.py`
    - Use `@given(confidence_scores=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=100), image_type=st.sampled_from([...]), strategy_name=st.sampled_from([...]))` with `@settings(max_examples=200)`
    - Assert `avg_confidence` is always in [0.0, 1.0] after any sequence of updates
    - **Validates: Requirements 7.3**

  - [ ]* 2.3 Write unit tests for Memory Manager
    - File: `tests/unit/test_memory_manager.py`
    - Test: corrupted LTM file → fresh init without crash (Req 7.4)
    - Test: `was_strategy_tried` returns True after `record_tried_strategy` (Req 6.3)
    - Test: `end_session` clears STM and writes log file (Req 6.5)
    - Test: `reset_long_term_memory` deletes file and reinitialises (Req 7.7)
    - Test: STM holds 100+ action records without error (Req 6.6)
    - _Requirements: 6.3, 6.5, 6.6, 7.4, 7.7_

- [ ] 3. Implement Agent Tools — wrap all Phase 1 components
  - [x] 3.1 Create `src/agents/tools/preprocessor_tool.py` — `PreprocessorTool(AgentTool)`
    - `_execute(inputs)`: calls `ImagePreprocessor().preprocess(inputs["image_path"])`, returns `{"preprocessed_path": ..., "original_path": ...}`
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [x] 3.2 Create `src/agents/tools/diagram_tool.py` — `DiagramTool(AgentTool)`
    - `_execute(inputs)`: calls `DiagramDetector().detect_and_extract(inputs["image_path"])`, returns the detector's result dict
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [x] 3.3 Create `src/agents/tools/ocr_tool.py` — `OCRTool(AgentTool)`
    - `_execute(inputs)`: reads `strategy_params["engine"]`, lazy-initialises `HybridOCR` per engine, calls `extract_text_from_image(image_path)`
    - `cleanup()`: releases all loaded OCR models
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [x] 3.4 Create `src/agents/tools/corrector_tool.py` — `CorrectorTool(AgentTool)`
    - `_execute(inputs)`: calls `LLMCorrector().correct_text(inputs["text"])` then `structure_content()`, returns `{"corrected_text": ..., "structured_text": ...}`
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [x] 3.5 Create `src/agents/tools/document_tool.py` — `DocumentTool(AgentTool)`
    - `_execute(inputs)`: calls `WordGenerator().create_document(content, title, diagrams=diagrams)`, returns `{"output_path": ...}`
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [ ]* 3.6 Write property test for Agent Tool always returns required fields (Property 7)
    - **Property 7: Agent Tool Always Returns Required Fields**
    - File: `tests/property/test_tool_interface_properties.py`
    - Use `@given(inputs=st.fixed_dictionaries({...}))` with arbitrary/malformed inputs for each tool
    - Assert every `invoke()` result contains `"status"`, `"output"`, `"execution_time_seconds"`
    - Assert when `status == "error"`, `"error_message"` is present
    - **Validates: Requirements 8.2, 8.3**

  - [ ]* 3.7 Write unit tests for tool wrappers
    - File: `tests/unit/tools/test_ocr_tool.py`, `test_preprocessor_tool.py`, `test_document_tool.py`
    - Test: each tool returns `status: "error"` (not exception) when given invalid image path
    - Test: `OCRTool.cleanup()` does not raise
    - _Requirements: 8.2, 8.3_

- [x] 4. Checkpoint — verify foundation and tools
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Implement Quality Assessment Agent
  - [x] 5.1 Create `src/agents/quality_agent.py` — `QualityAssessmentAgent(BaseAgent)`
    - `execute(inputs)`: computes 4 signals, weighted average, clamps to [0.0, 1.0], classifies, builds rationale, records to memory
    - `_signal_engine_confidence(engine_confidence)`: returns 0.5 when None
    - `_signal_text_coherence(text)`: word-length heuristic
    - `_signal_word_count(text)`: plausibility check (50–600 words = 0.9)
    - `_signal_garbled_sequences(text)`: regex-based penalty
    - `_compute_confidence(signals)`: weighted average (0.35/0.25/0.20/0.20)
    - `_classify(confidence_score)`: thresholds 0.8 / 0.5
    - `_build_rationale(signals, score, classification)`: plain-English string
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 17.4_

  - [ ]* 5.2 Write property test for confidence score always in valid range (Property 3)
    - **Property 3: Confidence Score Always in Valid Range**
    - File: `tests/property/test_confidence_score_properties.py`
    - Use `@given(text=st.text(min_size=0, max_size=5000), engine_confidence=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0)))` with `@settings(max_examples=200)`
    - Assert `0.0 <= result["output"]["confidence_score"] <= 1.0` for all inputs
    - **Validates: Requirements 5.1, 12.1**

  - [ ]* 5.3 Write property test for quality classification matches threshold (Property 4)
    - **Property 4: Quality Classification Matches Threshold**
    - File: `tests/property/test_confidence_score_properties.py`
    - Use `@given(confidence=st.floats(min_value=0.0, max_value=1.0))` with `@settings(max_examples=200)`
    - Call `agent._classify(confidence)` directly and assert the three threshold conditions
    - **Validates: Requirements 5.3, 5.4, 5.5**

  - [ ]* 5.4 Write property test for quality report always contains required fields (Property 9)
    - **Property 9: Quality Report Always Contains Required Fields**
    - File: `tests/property/test_confidence_score_properties.py`
    - Use `@given(text=st.text(), engine_confidence=st.one_of(st.none(), st.floats(0.0, 1.0)))` with `@settings(max_examples=200)`
    - Assert output dict always contains `"confidence_score"`, `"classification"`, `"rationale"`
    - **Validates: Requirements 5.6**

  - [ ]* 5.5 Write unit tests for Quality Assessment Agent
    - File: `tests/unit/test_quality_agent.py`
    - Test: empty text → `low_quality` classification
    - Test: coherent 200-word text → `confidence_score >= 0.5`
    - Test: text full of garbled sequences → lower score than clean text
    - Test: `execute()` completes within 5 seconds (Req 5.7)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

- [ ] 6. Implement Image Analysis Agent
  - [x] 6.1 Create `src/agents/image_analysis_agent.py` — `ImageAnalysisAgent(BaseAgent)`
    - `execute(inputs)`: loads image with OpenCV, extracts characteristics, classifies, builds `ImageAnalysisReport`, records to memory
    - `_estimate_resolution(img)`: A4-based DPI heuristic
    - `_estimate_noise_level(img)`: Laplacian variance normalised to [0, 1]
    - `_estimate_contrast(img)`: max/min pixel ratio
    - `_detect_diagrams_presence(img)`: Hough line detection (>10 lines → True)
    - `_classify_image_type(img, resolution, noise_level, has_diagrams)`: priority order — low_quality → printed_text → mixed_text_diagram → dense_handwriting
    - `_recommend_enhancements(resolution, noise_level, contrast_ratio)`: returns list of actionable strings
    - When a characteristic cannot be determined, record it as `"undetermined"` (Req 2.5)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 17.2_

  - [ ]* 6.2 Write unit tests for Image Analysis Agent
    - File: `tests/unit/test_image_analysis_agent.py`
    - Test: very small/low-res image → `image_type == "low_quality"` (Req 2.3)
    - Test: `execute()` returns all required fields: `image_type`, `resolution_dpi`, `noise_level`, `contrast_ratio`, `has_diagrams`, `recommended_enhancements` (Req 2.1)
    - Test: `execute()` completes within 10 seconds on CPU (Req 2.4)
    - Test: unreadable image path → `status: "error"` without crash
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [ ] 7. Implement Strategy Selection Agent
  - [x] 7.1 Create `src/agents/strategy_agent.py` — `StrategySelectionAgent(BaseAgent)` with `STRATEGY_REGISTRY`
    - `execute(inputs)`: filters strategies by API availability, scores each, sorts descending, returns top ≥3 candidates, generates explainability report, logs decision
    - `_get_available_strategies(api_available)`: filters `STRATEGY_REGISTRY` by `requires_api`
    - `_score_strategies(strategies, image_type, user_preference)`: base score (1.0 if in `best_for`, else 0.5) + memory score (40% weight when ≥5 records) + preference bonus (0.1)
    - `_build_rationale(strategy, base_score, memory_score, has_memory, user_preference)`: plain-English string
    - `_generate_explainability(selected, candidates, image_type, user_preference)`: includes bias disclosure statement
    - When API key absent, exclude `groq_api` from candidates (Req 3.5)
    - When fewer than 5 LTM records for image_type, apply equal prior (Req 3.3, 14.3)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 9.2, 14.1, 14.2, 14.3, 17.3_

  - [ ]* 7.2 Write property test for strategy selection always produces minimum candidates (Property 8)
    - **Property 8: Strategy Selection Always Produces Minimum Candidates**
    - File: `tests/property/test_strategy_selection_properties.py`
    - Use `@given(image_type=st.sampled_from([...]), api_available=st.booleans())` with `@settings(max_examples=200)`
    - Assert `len(result["output"]["candidate_strategies"]) >= 3`
    - **Validates: Requirements 3.4**

  - [ ]* 7.3 Write property test for equal prior when memory is insufficient (Property 10)
    - **Property 10: Equal Prior When Memory Is Insufficient**
    - File: `tests/property/test_strategy_selection_properties.py`
    - Use `@given(image_type=st.sampled_from([...]))` with fresh `MemoryManager` (0 records)
    - Score all strategies and assert all base scores are equal (0.5) before preference bonus
    - **Validates: Requirements 14.3, 3.3**

  - [ ]* 7.4 Write unit tests for Strategy Selection Agent
    - File: `tests/unit/test_strategy_agent.py`
    - Test: `api_available=False` → no `groq_api` in candidates (Req 3.5)
    - Test: user preference engine appears in candidates and is noted in explainability (Req 3.7)
    - Test: explainability report does not contain "neural network weights", "embedding" (Req 9.6)
    - Test: `execute()` completes within 3 seconds (Req 17.3)
    - Test: with ≥5 LTM records, memory weight ≥40% influences selection (Req 3.3)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

- [x] 8. Checkpoint — verify all agents individually
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Implement Orchestrator Agent
  - [x] 9.1 Create `src/agents/orchestrator_agent.py` — `OrchestratorAgent(BaseAgent)`
    - `__init__(memory_manager, tools)`: instantiates `ImageAnalysisAgent`, `StrategySelectionAgent`, `QualityAssessmentAgent`
    - `execute(inputs)`: runs the full Perceive → Decide → Act → Learn loop
      - Phase 1 (Perceive): calls `image_analysis_agent.execute()`
      - Phase 2 (Decide): calls `strategy_agent.execute()` or uses `user_override`
      - Phase 3 (Act): retry loop up to `RETRY_BUDGET=3`; calls `_execute_strategy()` then `quality_agent.execute()`; tracks best result; breaks when `confidence >= 0.5`
      - Triggers HITL when all retries exhausted and best confidence < 0.5
      - Phase 4 (Learn): calls `memory_manager.update_long_term_memory()`
      - Records every retry attempt, strategy, and confidence in Decision Log (Req 4.5)
    - `_execute_strategy(strategy, image_path)`: looks up tool by `strategy["tool_name"]`, calls `tool.invoke()`
    - `_build_error_result(message, error_details)`: standardised error dict
    - Catches all unhandled exceptions from tools and agents, logs to Decision Log, proceeds to next strategy (Req 4.6)
    - Calls Phase 1 components exclusively through tool wrappers (Req 8.4)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 18.5_

  - [ ]* 9.2 Write property test for retry budget never exceeded (Property 2)
    - **Property 2: Retry Budget Never Exceeded**
    - File: `tests/property/test_retry_budget_properties.py`
    - Use `@given(confidence_scores=st.lists(st.floats(min_value=0.0, max_value=0.49), min_size=1, max_size=10))` with `@settings(max_examples=100)`
    - Mock tools to return OCR results with the given below-threshold confidence scores
    - Assert `result["retry_count"] <= 3`
    - **Validates: Requirements 4.3**

  - [ ]* 9.3 Write property test for agentic loop phase ordering (Property 1)
    - **Property 1: Agentic Loop Phase Ordering**
    - File: `tests/property/test_retry_budget_properties.py`
    - Use `@given(st.just("test_image.jpg"))` with mocked sub-agents
    - Assert `phases_completed` is always a prefix of `["perceive", "decide", "act", "learn"]` — no phase appears before its predecessor
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**

  - [ ]* 9.4 Write property test for strategy deduplication within session (Property 5)
    - **Property 5: Strategy Deduplication Within Session**
    - File: `tests/property/test_retry_budget_properties.py`
    - Use `@given(strategy_names=st.lists(st.sampled_from(["groq_api", "got_ocr_local", "florence_local", "easyocr_local"]), min_size=1, max_size=10))`
    - Call `record_tried_strategy` for each name and assert `tried_strategies` contains no duplicates
    - **Validates: Requirements 6.3**

  - [ ]* 9.5 Write unit tests for Orchestrator Agent
    - File: `tests/unit/test_orchestrator_agent.py`
    - Test: all retries exhausted → `hitl_triggered == True` (Req 4.4)
    - Test: first strategy succeeds (confidence ≥ 0.5) → `retry_count == 0` (Req 4.1)
    - Test: tool raises exception → orchestrator catches it, logs it, tries next strategy (Req 4.6)
    - Test: `phases_completed` contains all four phases on success (Req 1.6)
    - Test: user override bypasses strategy agent (Req 10.4)
    - _Requirements: 1.1, 1.6, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [x] 10. Wire agents and tools together — create the agent factory
  - Create `src/agents/__init__.py` with a `create_agentic_system() -> OrchestratorAgent` factory function
    - Instantiates `MemoryManager`
    - Instantiates all tools: `PreprocessorTool`, `DiagramTool`, `OCRTool`, `CorrectorTool`, `DocumentTool`
    - Instantiates `OrchestratorAgent(memory_manager, tools)`
    - Returns the orchestrator
  - Verify the factory can be imported and called without errors
  - _Requirements: 19.3, 19.4_

- [x] 11. Checkpoint — verify full agent stack
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 12. Implement the Agentic Streamlit Dashboard (`agentic_app.py`)
  - [x] 12.1 Create `agentic_app.py` with the agentic dashboard layout
    - Page config, CSS, session state initialisation (session_id as UUID, orchestrator instance, processing state)
    - Sidebar: Full Override Mode toggle (radio: Agentic / Manual), LTM stats panel (per image_type: record count + best strategy), API status, Reset Memory button with confirmation
    - Main area: image upload, output name input, Override controls (OCR engine selector + preprocessing checkbox, visible when override active)
    - Phase indicator bar: Perceive → Decide → Act → Learn with colour coding (updates via `st.empty()` placeholders)
    - _Requirements: 20.1, 20.4, 20.5, 10.3, 15.5, 15.6_

  - [x] 12.2 Implement agentic processing flow in `agentic_app.py`
    - On "Process Image" button click: generate UUID session_id, call `orchestrator.execute()` with image_path and optional user_override
    - Update phase indicator in real time as each phase completes (use `st.status` or `st.empty` updates)
    - After processing: display Explainability Panel (expandable) with image_type, selected strategy, rank, top-3 reasons, retry history, bias disclosure (Req 9.1, 9.2, 9.3, 9.5, 9.6)
    - Display Confidence Score as percentage + quality classification badge (Req 12.1, 12.2)
    - Show warning when classification is `low_quality` (Req 12.3)
    - Display Short-Term Memory timeline (collapsible, agent actions with timestamps) (Req 20.3)
    - Editable text area for extracted text before document generation (Req 10.5)
    - Download buttons: Word Document, Decision Log JSON, Session Memory JSON (Req 9.4)
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 12.1, 12.2, 12.3, 20.1, 20.2, 20.3_

  - [x] 12.3 Implement Human-in-the-Loop dialog in `agentic_app.py`
    - When `result["hitl_triggered"] == True`, display a blocking confirmation section (using `st.warning` + `st.radio` + `st.button`) showing best result preview, confidence score, and three choices: Accept / Choose Strategy Manually / Cancel
    - When "Choose Strategy Manually" selected, show strategy selector and re-run with override
    - When HITL is active, block document generation until user responds (Req 10.2)
    - _Requirements: 4.4, 10.1, 10.2, 20.6_

  - [x] 12.4 Implement Proactive Suggestions in `agentic_app.py`
    - After image analysis completes (before OCR), check `analysis_report["image_type"]` and `noise_level`
    - If `image_type == "low_quality"`: display suggestion with specific recommendation from `recommended_enhancements` (Req 11.1, 11.2)
    - If `noise_level > 0.6`: offer "Apply automatic denoising" checkbox and await confirmation before proceeding (Req 11.3)
    - Display suggestions before OCR begins (Req 11.4)
    - Include "Dismiss" button for each suggestion (Req 11.5)
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

  - [x] 12.5 Implement Full Override Mode (Phase 1 fallback) in `agentic_app.py`
    - When Full Override Mode is active, display red banner: "⚠️ Full Override Mode Active — Autonomous decisions disabled."
    - In override mode, bypass all agents and call Phase 1 components directly (matching `app.py` behaviour)
    - Record mode activation in Decision Log (Req 15.6)
    - _Requirements: 15.5, 15.6, 20.4_

  - [x] 12.6 Implement privacy notices and terms-of-use in `agentic_app.py`
    - On first launch (check `st.session_state`), display terms-of-use notice covering data handling and legal constraints (Req 16.4)
    - Before first Groq API call, display notice: "Image data will be sent to the Groq API" (Req 13.2)
    - _Requirements: 13.1, 13.2, 16.4_

- [x] 13. Implement session cleanup and privacy safeguards
  - Add `cleanup_session(session_id)` logic called at end of each processing run:
    - Delete temporary files from `uploads/` and `temp/` created during the session (Req 13.3)
    - Call `memory_manager.end_session(session_id)` to persist log and clear STM (Req 6.5)
  - Verify `memory_manager.end_session()` clears STM to prevent data leakage (Req 6.5)
  - Add "Delete All Stored Data" button in sidebar that calls `memory_manager.reset_long_term_memory()` and deletes session logs (Req 13.6)
  - _Requirements: 6.5, 13.3, 13.5, 13.6_

- [x] 14. Checkpoint — verify UI and privacy features
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 15. Write integration tests
  - [-] 15.1 Create `tests/integration/test_full_pipeline.py`
    - Test: upload a real test image, run full agentic loop via `OrchestratorAgent.execute()`, verify `status == "success"`, `phases_completed == ["perceive", "decide", "act", "learn"]`, and a `.docx` file is produced
    - Test: `execution_time_seconds <= 120` on CPU (Req 17.1)
    - _Requirements: 1.1, 1.6, 4.7, 17.1_

  - [-] 15.2 Create `tests/integration/test_api_fallback.py`
    - Test: set `GROQ_API_KEY` to empty string, run orchestrator, verify `groq_api` strategy is not selected and a local strategy is used (Req 3.5, 18.1)
    - Test: mock all local models to fail, verify EasyOCR is used as last resort (Req 18.2)
    - _Requirements: 3.5, 18.1, 18.2_

  - [-] 15.3 Create `tests/integration/test_memory_persistence.py`
    - Test: run two sequential sessions, verify LTM is updated with records from both sessions (Req 7.2)
    - Test: corrupt `memory/long_term_memory.json`, create new `MemoryManager`, verify it initialises fresh without crash (Req 7.4, 18.4)
    - _Requirements: 7.1, 7.2, 7.4, 18.4_

- [~] 16. Final checkpoint — full test suite
  - Ensure all tests pass, ask the user if questions arise.

---

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Phase 1 files (`src/preprocessing/`, `src/ocr/`, `src/postprocessing/`, `src/document_generation/`, `app.py`) must not be modified
- Property tests use **Hypothesis** (`pip install hypothesis`) and run with `pytest tests/property/ --tb=short`
- Unit tests run with `pytest tests/unit/ --tb=short`
- Integration tests require a real image file; place a sample in `tests/fixtures/sample_page.jpg`
- Each property test is tagged with a comment: `# Feature: agentic-ocr-pipeline, Property N: <Title>`
- The `agentic_app.py` is launched with `streamlit run agentic_app.py` (separate from `app.py`)
- Session IDs are generated with `uuid.uuid4()` at the start of each processing run
- The `memory/` and `logs/` directories should be added to `.gitignore` (except `.gitkeep`)
