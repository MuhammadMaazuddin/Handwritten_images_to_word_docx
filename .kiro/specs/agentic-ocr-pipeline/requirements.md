# Requirements Document

## Introduction

This document specifies the requirements for transforming the existing Phase 1 "Handwriting to Word Converter" — a static, sequential OCR pipeline — into a Purely Agentic System (Phase 2). The transformed system, referred to as the **Agentic OCR Pipeline**, introduces autonomous decision-making, self-evaluation, memory, and learning capabilities. It replaces the fixed, user-driven pipeline with an intelligent agent loop that observes image characteristics, selects optimal OCR strategies, evaluates output quality, recovers from errors autonomously, and continuously improves through historical performance data.

The system is designed for a university course context (Professional Practices in IT) and must comply with relevant ethical, legal, and professional standards including ACM/IEEE Code of Ethics, GDPR principles, and PECA 2016 awareness.

---

## Glossary

- **Agentic_System**: The complete Phase 2 system comprising all agents, memory, tools, and the Streamlit UI.
- **Orchestrator_Agent**: The central decision-making agent that coordinates all other agents and manages the processing loop.
- **Image_Analysis_Agent**: The agent responsible for analysing image characteristics (quality, content type, noise level, etc.) to inform strategy selection.
- **Strategy_Selection_Agent**: The agent that selects the optimal OCR strategy based on image analysis results and historical performance data.
- **Quality_Assessment_Agent**: The agent that evaluates OCR output quality and decides whether to accept, retry, or escalate to the user.
- **Memory_Manager**: The component that manages short-term (session) and long-term (persistent) memory for the Agentic_System.
- **Agent_Tool**: A callable wrapper around an existing Phase 1 component (e.g., HybridOCR, ImagePreprocessor) that an agent can invoke as an action.
- **OCR_Engine**: Any of the four OCR backends — Groq Llama Vision API, Florence-2, GOT-OCR 2.0, or EasyOCR.
- **Confidence_Score**: A numeric value in the range [0.0, 1.0] representing the Quality_Assessment_Agent's estimated reliability of an OCR result.
- **Strategy**: A named configuration specifying which OCR_Engine to use, whether to apply preprocessing, and any engine-specific parameters.
- **Short_Term_Memory**: In-session storage of processing steps, decisions, and intermediate results for the current user session.
- **Long_Term_Memory**: Persistent storage of historical performance data, user preferences, and learned strategy-to-image-type mappings across sessions.
- **Decision_Log**: A structured record of every decision made by an agent, including the rationale, inputs considered, and outcome.
- **Human_In_The_Loop**: A control mechanism that pauses autonomous processing and requests explicit user confirmation before proceeding.
- **Explainability_Report**: A human-readable summary of why the Agentic_System made a particular decision during processing.
- **Override**: A user-initiated action that replaces an agent's autonomous decision with a user-specified alternative.
- **PECA_2016**: Pakistan Electronic Crimes Act 2016 — the applicable cybercrime legislation for the deployment context.
- **GDPR_Mindset**: Application of data minimisation, purpose limitation, and user consent principles inspired by the EU General Data Protection Regulation.
- **ACM_IEEE_Ethics**: The ACM Code of Ethics and Professional Conduct and IEEE Code of Ethics, used as the professional ethical framework.
- **Retry_Budget**: The maximum number of autonomous retry attempts the Orchestrator_Agent may perform before escalating to the user.
- **Image_Type**: A classification of the input image, e.g., "dense_handwriting", "mixed_text_diagram", "formula_heavy", "low_quality", "printed_text".

---

## Requirements

---

### Requirement 1: Agentic Architecture — Perception–Decision–Action–Learning Loop

**User Story:** As a university assessor, I want the system to implement a complete agentic loop (Perceive → Interpret → Decide → Act → Learn), so that it qualifies as a Purely Agentic System under the course criteria.

#### Acceptance Criteria

1. THE Agentic_System SHALL implement a processing loop consisting of four sequential phases: Perception, Decision, Action, and Learning, executed in that order for every image submitted.
2. WHEN an image is submitted, THE Image_Analysis_Agent SHALL execute the Perception phase by extracting image characteristics before any OCR is attempted.
3. WHEN the Perception phase completes, THE Strategy_Selection_Agent SHALL execute the Decision phase by selecting a Strategy before invoking any OCR_Engine.
4. WHEN the Decision phase completes, THE Orchestrator_Agent SHALL execute the Action phase by invoking the selected Strategy via the appropriate Agent_Tool.
5. WHEN the Action phase completes, THE Memory_Manager SHALL execute the Learning phase by recording the outcome and updating Long_Term_Memory.
6. THE Agentic_System SHALL complete the full loop for every processed image, including images that result in errors.

---

### Requirement 2: Image Analysis Agent

**User Story:** As a developer, I want an Image Analysis Agent that characterises input images, so that downstream agents can make informed strategy decisions without manual user input.

#### Acceptance Criteria

1. WHEN an image is submitted, THE Image_Analysis_Agent SHALL analyse the image and produce a structured analysis report containing at minimum: estimated Image_Type, resolution, noise level, contrast ratio, and presence of diagrams.
2. THE Image_Analysis_Agent SHALL classify each image into exactly one Image_Type from the defined set: "dense_handwriting", "mixed_text_diagram", "formula_heavy", "low_quality", "printed_text".
3. WHEN image resolution is below 150 DPI equivalent, THE Image_Analysis_Agent SHALL flag the image as "low_quality" and include a recommended enhancement action in the analysis report.
4. THE Image_Analysis_Agent SHALL complete image analysis within 10 seconds on standard CPU hardware.
5. WHEN the Image_Analysis_Agent cannot determine a characteristic with sufficient confidence, THE Image_Analysis_Agent SHALL record that characteristic as "undetermined" rather than guessing.
6. THE Image_Analysis_Agent SHALL use the Groq Llama Vision API as its primary analysis backend, with a local OpenCV-based fallback for when the API is unavailable.

---

### Requirement 3: Strategy Selection Agent

**User Story:** As a user, I want the system to automatically select the best OCR strategy for my image, so that I do not need to manually choose an OCR engine.

#### Acceptance Criteria

1. WHEN an image analysis report is available, THE Strategy_Selection_Agent SHALL select a Strategy without requiring user input.
2. THE Strategy_Selection_Agent SHALL consider the Image_Type, noise level, and Long_Term_Memory performance data when selecting a Strategy.
3. WHEN Long_Term_Memory contains at least 5 historical records for the identified Image_Type, THE Strategy_Selection_Agent SHALL weight historical success rates at no less than 40% of the selection decision.
4. THE Strategy_Selection_Agent SHALL produce a ranked list of at least 3 candidate Strategies, ordered by predicted success probability, and record this list in the Decision_Log.
5. WHEN the Groq API key is absent or the API is unreachable, THE Strategy_Selection_Agent SHALL exclude API-dependent Strategies from the candidate list and select from local Strategies only.
6. THE Strategy_Selection_Agent SHALL generate an Explainability_Report for every selection decision, stating the top 3 reasons for the chosen Strategy in plain English.
7. WHERE the user has set a preferred OCR_Engine in their preferences, THE Strategy_Selection_Agent SHALL include that engine's Strategy as a candidate and note the user preference in the Explainability_Report.

---

### Requirement 4: Orchestrator Agent — Autonomous Execution and Retry

**User Story:** As a user, I want the system to autonomously retry failed OCR attempts with different strategies, so that I do not need to manually intervene on every failure.

#### Acceptance Criteria

1. THE Orchestrator_Agent SHALL invoke the Strategy selected by the Strategy_Selection_Agent as the first OCR attempt.
2. WHEN the Quality_Assessment_Agent returns a Confidence_Score below 0.5 for an OCR result, THE Orchestrator_Agent SHALL autonomously retry using the next-ranked Strategy from the candidate list.
3. THE Orchestrator_Agent SHALL not exceed the Retry_Budget of 3 autonomous retry attempts per image.
4. WHEN all retry attempts are exhausted and no result has a Confidence_Score of 0.5 or above, THE Orchestrator_Agent SHALL trigger the Human_In_The_Loop mechanism and present the best available result to the user for review.
5. THE Orchestrator_Agent SHALL record every retry attempt, the Strategy used, and the resulting Confidence_Score in the Decision_Log.
6. WHEN an Agent_Tool raises an unhandled exception, THE Orchestrator_Agent SHALL catch the exception, log it to the Decision_Log, and proceed to the next Strategy without crashing.
7. THE Orchestrator_Agent SHALL complete the full processing loop, including up to 3 retries, within 120 seconds for a standard A4-sized image on CPU hardware.

---

### Requirement 5: Quality Assessment Agent

**User Story:** As a user, I want the system to evaluate the quality of its own OCR output, so that I receive only reliable results and am alerted when quality is insufficient.

#### Acceptance Criteria

1. WHEN an OCR result is produced, THE Quality_Assessment_Agent SHALL evaluate it and return a Confidence_Score in the range [0.0, 1.0].
2. THE Quality_Assessment_Agent SHALL compute the Confidence_Score using at least three of the following signals: character-level confidence from the OCR_Engine, text coherence (language model perplexity proxy), expected word count vs. actual, and presence of garbled character sequences.
3. WHEN the Confidence_Score is 0.8 or above, THE Quality_Assessment_Agent SHALL classify the result as "high_quality".
4. WHEN the Confidence_Score is between 0.5 (inclusive) and 0.8 (exclusive), THE Quality_Assessment_Agent SHALL classify the result as "acceptable".
5. WHEN the Confidence_Score is below 0.5, THE Quality_Assessment_Agent SHALL classify the result as "low_quality" and recommend a retry.
6. THE Quality_Assessment_Agent SHALL include the Confidence_Score, quality classification, and a brief rationale in every evaluation report.
7. THE Quality_Assessment_Agent SHALL complete evaluation within 5 seconds per OCR result.

---

### Requirement 6: Memory Manager — Short-Term Memory

**User Story:** As a user, I want the system to remember what it has done within my current session, so that it can avoid repeating failed strategies and provide a coherent processing history.

#### Acceptance Criteria

1. THE Memory_Manager SHALL maintain a Short_Term_Memory store for the duration of each user session.
2. WHEN any agent completes an action, THE Memory_Manager SHALL record the action type, inputs, outputs, timestamp, and Confidence_Score (if applicable) in Short_Term_Memory.
3. WHEN the Orchestrator_Agent is about to retry with a Strategy, THE Memory_Manager SHALL check Short_Term_Memory and prevent the same Strategy from being retried on the same image within the same session.
4. THE Memory_Manager SHALL make the full Short_Term_Memory log available to the user as a downloadable JSON file at the end of each session.
5. WHEN a user session ends, THE Memory_Manager SHALL clear Short_Term_Memory to prevent data leakage between sessions.
6. THE Short_Term_Memory store SHALL hold at least 100 action records per session without performance degradation.

---

### Requirement 7: Memory Manager — Long-Term Memory and Learning

**User Story:** As a user, I want the system to learn from past processing results, so that it improves its strategy selection over time.

#### Acceptance Criteria

1. THE Memory_Manager SHALL persist Long_Term_Memory to a local JSON file at the end of every session.
2. WHEN a processing loop completes, THE Memory_Manager SHALL update the Long_Term_Memory record for the corresponding Image_Type and Strategy with the final Confidence_Score and success/failure outcome.
3. THE Memory_Manager SHALL maintain a per-Strategy, per-Image_Type success rate calculated as a rolling average over the last 50 records.
4. WHEN the Long_Term_Memory file is absent or corrupted, THE Memory_Manager SHALL initialise a new empty Long_Term_Memory store and log a warning without crashing.
5. THE Memory_Manager SHALL not store any image pixel data, file paths containing personally identifiable information, or raw OCR text in Long_Term_Memory.
6. THE Long_Term_Memory file SHALL be stored locally on the user's machine and SHALL NOT be transmitted to any external server.
7. WHERE the user requests a reset of learned preferences, THE Memory_Manager SHALL delete the Long_Term_Memory file and confirm the reset to the user.

---

### Requirement 8: Agent Tools — Wrapping Phase 1 Components

**User Story:** As a developer, I want all Phase 1 pipeline components to be wrapped as Agent Tools, so that agents can invoke them through a uniform interface without direct coupling.

#### Acceptance Criteria

1. THE Agentic_System SHALL provide an Agent_Tool wrapper for each of the following Phase 1 components: ImagePreprocessor, DiagramDetector, HybridOCR (per engine), LLMCorrector, and WordGenerator.
2. EACH Agent_Tool SHALL expose a uniform `invoke(inputs: dict) -> dict` interface returning at minimum: `status` ("success" or "error"), `output`, and `execution_time_seconds`.
3. WHEN an Agent_Tool invocation fails, THE Agent_Tool SHALL return `status: "error"` with a descriptive `error_message` field rather than raising an unhandled exception.
4. THE Orchestrator_Agent SHALL invoke Phase 1 components exclusively through Agent_Tool wrappers and SHALL NOT call Phase 1 classes directly.
5. EACH Agent_Tool SHALL log its invocation, inputs (excluding raw image data), and result to the Decision_Log.

---

### Requirement 9: Explainability and Decision Transparency

**User Story:** As a user, I want to understand why the system chose a particular OCR method and how confident it is, so that I can trust the output and make informed corrections.

#### Acceptance Criteria

1. THE Agentic_System SHALL display an Explainability_Report to the user in the Streamlit UI after every processing run.
2. THE Explainability_Report SHALL include: the detected Image_Type, the selected Strategy and its rank in the candidate list, the Confidence_Score of the final result, the number of retries performed, and a plain-English rationale for the strategy choice.
3. WHEN the system retried one or more times, THE Explainability_Report SHALL list each attempted Strategy and its Confidence_Score in chronological order.
4. THE Agentic_System SHALL make the full Decision_Log available for download as a JSON file from the Streamlit UI.
5. THE Explainability_Report SHALL be written in plain English at a level understandable to a non-technical user.
6. THE Agentic_System SHALL NOT use the terms "neural network weights", "embedding", or other internal ML jargon in the user-facing Explainability_Report.

---

### Requirement 10: Human-in-the-Loop Controls

**User Story:** As a user, I want to be able to override the system's autonomous decisions and maintain control over the final output, so that I am not fully dependent on the agent's judgement.

#### Acceptance Criteria

1. THE Agentic_System SHALL provide a Human_In_The_Loop confirmation step when the Orchestrator_Agent exhausts its Retry_Budget without achieving a Confidence_Score of 0.5 or above.
2. WHEN the Human_In_The_Loop step is triggered, THE Agentic_System SHALL present the user with the best available OCR result, its Confidence_Score, and a choice to: accept the result, manually select an alternative Strategy, or cancel processing.
3. THE Agentic_System SHALL provide an Override control in the Streamlit UI that allows the user to manually specify the OCR_Engine and preprocessing options at any time before processing begins.
4. WHEN the user activates an Override, THE Strategy_Selection_Agent SHALL record the override in the Decision_Log and use the user-specified Strategy instead of its autonomous selection.
5. THE Agentic_System SHALL allow the user to edit the extracted text in the Streamlit UI before generating the Word document.
6. THE Agentic_System SHALL provide a "Pause Agent" control that halts autonomous processing after the current step and awaits user instruction.

---

### Requirement 11: Proactive Suggestions

**User Story:** As a user, I want the system to proactively suggest improvements to my image before processing, so that I can get better results without needing OCR expertise.

#### Acceptance Criteria

1. WHEN the Image_Analysis_Agent detects a "low_quality" Image_Type, THE Agentic_System SHALL display a proactive suggestion to the user recommending image enhancement before proceeding.
2. THE proactive suggestion SHALL include a specific, actionable recommendation (e.g., "Increase brightness", "Scan at higher resolution", "Remove shadows") derived from the image analysis report.
3. WHEN the Image_Analysis_Agent detects a noise level above 0.6 (on a normalised 0–1 scale), THE Agentic_System SHALL offer to apply automatic denoising preprocessing and await user confirmation before applying it.
4. THE Agentic_System SHALL display proactive suggestions before OCR begins, not after.
5. THE Agentic_System SHALL allow the user to dismiss any proactive suggestion and proceed without acting on it.

---

### Requirement 12: Confidence Scores and Quality Reporting

**User Story:** As a user, I want to see confidence scores and quality assessments for the OCR output, so that I know how much to trust the converted text.

#### Acceptance Criteria

1. THE Agentic_System SHALL display the final Confidence_Score as a percentage in the Streamlit UI results section after every processing run.
2. THE Agentic_System SHALL display the quality classification ("high_quality", "acceptable", or "low_quality") alongside the Confidence_Score.
3. WHEN the quality classification is "low_quality", THE Agentic_System SHALL display a warning message advising the user to review the extracted text carefully.
4. THE Agentic_System SHALL display per-section confidence scores when the image contains multiple distinct text regions.
5. THE Confidence_Score displayed to the user SHALL be the score of the final accepted OCR result, not an average across retries.

---

### Requirement 13: Privacy and Data Protection

**User Story:** As a user, I want my uploaded images and extracted text to be handled with privacy by default, so that my personal or sensitive notes are not exposed or retained without my consent.

#### Acceptance Criteria

1. THE Agentic_System SHALL process all uploaded images locally on the user's machine and SHALL NOT transmit image data to any external server except when the user has explicitly selected the Groq API OCR_Engine.
2. WHEN the Groq API OCR_Engine is used, THE Agentic_System SHALL display a notice to the user stating that image data will be sent to the Groq API before the first API call is made.
3. THE Agentic_System SHALL delete all temporary image files (preprocessed images, diagram extracts) from the local filesystem at the end of each processing session.
4. THE Memory_Manager SHALL store only aggregated performance metrics in Long_Term_Memory and SHALL NOT store raw OCR text, image thumbnails, or file names that could identify the user's content.
5. THE Agentic_System SHALL not collect, transmit, or log any user-identifying information (name, email, device ID) without explicit user consent.
6. WHERE the user requests deletion of all stored data, THE Agentic_System SHALL delete the Long_Term_Memory file and all session logs within the same session.

---

### Requirement 14: Ethical Agent Design — Bias and Fairness

**User Story:** As a course assessor, I want the system to address potential bias in its AI decision-making, so that the design demonstrates awareness of ACM/IEEE ethical obligations.

#### Acceptance Criteria

1. THE Strategy_Selection_Agent SHALL not use any demographic, linguistic, or cultural attributes of the user as inputs to strategy selection decisions.
2. THE Agentic_System SHALL include a bias disclosure statement in the Explainability_Report noting that strategy selection is based solely on image characteristics and historical performance data.
3. WHEN Long_Term_Memory contains fewer than 5 records for an Image_Type, THE Strategy_Selection_Agent SHALL apply equal prior probability to all candidate Strategies rather than defaulting to a potentially biased historical preference.
4. THE Agentic_System SHALL log all strategy selection decisions in the Decision_Log to enable post-hoc bias auditing.
5. THE Agentic_System documentation SHALL include a section identifying known limitations and potential sources of bias in the OCR models used (e.g., language bias in EasyOCR, training data limitations of Florence-2).

---

### Requirement 15: Safety Mechanisms — Logging and Override

**User Story:** As a developer and course assessor, I want the system to maintain comprehensive logs and support full override capability, so that the agent's behaviour can be audited and corrected.

#### Acceptance Criteria

1. THE Agentic_System SHALL maintain a structured Decision_Log for every processing run, stored as a JSON file in a local `logs/` directory.
2. THE Decision_Log SHALL record: session ID, timestamp, image filename (without path), Image_Type, all candidate Strategies with scores, selected Strategy, retry history, final Confidence_Score, and any Human_In_The_Loop events.
3. THE Agentic_System SHALL retain Decision_Log files for the duration of the application's runtime and SHALL NOT automatically delete them.
4. WHEN an unhandled exception occurs in any agent, THE Agentic_System SHALL log the full stack trace to the Decision_Log and display a user-friendly error message in the Streamlit UI without exposing internal stack traces to the user.
5. THE Agentic_System SHALL provide a "Full Override Mode" in the Streamlit UI sidebar that disables all autonomous agent decisions and reverts to the Phase 1 manual pipeline behaviour.
6. WHEN Full Override Mode is active, THE Agentic_System SHALL display a clear visual indicator in the UI and record the mode activation in the Decision_Log.

---

### Requirement 16: Legal and Professional Compliance

**User Story:** As a course assessor, I want the system design to demonstrate awareness of applicable legal and professional obligations, so that the project meets the Professional Practices in IT course requirements.

#### Acceptance Criteria

1. THE Agentic_System documentation SHALL include an IPR section identifying the licences of all third-party libraries and models used (including Groq API terms, Florence-2 MIT licence, GOT-OCR 2.0 Apache 2.0, EasyOCR Apache 2.0, python-docx MIT).
2. THE Agentic_System documentation SHALL include a PECA 2016 awareness section noting that the system must not be used to process, store, or transmit content that violates Pakistani cybercrime law.
3. THE Agentic_System documentation SHALL include an ACM/IEEE Code of Ethics compliance section mapping at least 5 ethical principles to specific design decisions in the system.
4. THE Agentic_System SHALL display a terms-of-use notice on first launch informing the user of data handling practices and applicable legal constraints.
5. WHERE the Groq API is used, THE Agentic_System SHALL comply with Groq's API terms of service, including not storing API responses beyond the current session.

---

### Requirement 17: Non-Functional — Performance

**User Story:** As a user, I want the agentic system to process my images within a reasonable time, so that the added intelligence does not make the tool impractically slow.

#### Acceptance Criteria

1. THE Agentic_System SHALL complete the full agentic loop (analysis, strategy selection, OCR, quality assessment, document generation) for a single standard A4 image within 120 seconds on CPU-only hardware.
2. THE Image_Analysis_Agent SHALL complete image analysis within 10 seconds.
3. THE Strategy_Selection_Agent SHALL complete strategy selection within 3 seconds.
4. THE Quality_Assessment_Agent SHALL complete quality evaluation within 5 seconds per OCR result.
5. THE Memory_Manager SHALL complete Long_Term_Memory read and write operations within 1 second.
6. WHEN GPU hardware is available, THE Agentic_System SHALL utilise it for local model inference and SHALL complete the full loop within 60 seconds.

---

### Requirement 18: Non-Functional — Reliability and Graceful Degradation

**User Story:** As a user, I want the system to remain functional even when some components are unavailable, so that I can always get some output from my uploaded image.

#### Acceptance Criteria

1. WHEN the Groq API is unavailable, THE Agentic_System SHALL automatically fall back to local OCR_Engines without user intervention and SHALL notify the user of the fallback.
2. WHEN all local models fail to load, THE Agentic_System SHALL fall back to EasyOCR as the last-resort OCR_Engine.
3. WHEN EasyOCR also fails, THE Agentic_System SHALL inform the user that OCR is unavailable and provide diagnostic guidance rather than crashing.
4. WHEN the Long_Term_Memory file is corrupted or unreadable, THE Agentic_System SHALL initialise a fresh memory store and continue processing without crashing.
5. THE Agentic_System SHALL remain operational (able to accept and process images) even if the Memory_Manager, Strategy_Selection_Agent, or Quality_Assessment_Agent individually fail, by falling back to the Phase 1 sequential pipeline behaviour.

---

### Requirement 19: Non-Functional — Maintainability and Extensibility

**User Story:** As a developer, I want the agentic architecture to be modular and extensible, so that new OCR engines or agents can be added without restructuring the entire system.

#### Acceptance Criteria

1. THE Agentic_System SHALL define a base `Agent` abstract class that all agents (Orchestrator_Agent, Image_Analysis_Agent, Strategy_Selection_Agent, Quality_Assessment_Agent) inherit from.
2. THE Agentic_System SHALL define a base `AgentTool` abstract class that all Agent_Tool wrappers implement.
3. WHEN a new OCR_Engine is added, THE Agentic_System SHALL require changes only to the Agent_Tool layer and the Strategy_Selection_Agent's strategy registry, with no changes required to the Orchestrator_Agent or Quality_Assessment_Agent.
4. THE Agentic_System SHALL include inline docstrings for all agent classes and Agent_Tool wrappers describing their inputs, outputs, and decision logic.
5. THE Agentic_System SHALL pass all existing Phase 1 unit tests without modification, ensuring backward compatibility.

---

### Requirement 20: Streamlit UI — Agentic Dashboard

**User Story:** As a user, I want the Streamlit interface to reflect the agentic nature of the system, so that I can observe the agent's reasoning and interact with it meaningfully.

#### Acceptance Criteria

1. THE Agentic_System SHALL update the Streamlit UI in real time as each agent phase (Perception, Decision, Action, Learning) completes, displaying the current phase name and status.
2. THE Agentic_System SHALL display the Explainability_Report in a dedicated expandable section of the Streamlit UI after processing completes.
3. THE Agentic_System SHALL display the Short_Term_Memory log as a collapsible timeline in the Streamlit UI, showing each agent action with its timestamp and outcome.
4. THE Agentic_System SHALL provide a sidebar toggle labelled "Full Override Mode" that switches between agentic and manual (Phase 1) pipeline behaviour.
5. THE Agentic_System SHALL display the current Long_Term_Memory statistics (number of records per Image_Type, top-performing Strategy per type) in a sidebar panel.
6. WHEN the Human_In_The_Loop step is triggered, THE Agentic_System SHALL display a modal-style confirmation dialog in the Streamlit UI that blocks further processing until the user responds.
