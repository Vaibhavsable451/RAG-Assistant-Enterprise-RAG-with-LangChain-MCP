"""
Streamlit Front End for RAG Assistant & AI Governance Suite.
Handles PDF/TXT document indexing with persistent session state across chat reruns,
and provides a 7-tab AI Governance Management Dashboard.
"""
import os
import json
import time
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from config import settings
from rag.chunking import chunk_documents
from rag.vector_store import build_vector_store, load_vector_store, add_to_vector_store, clear_vector_store
from rag.retriever import get_relevant_chunks, format_context
from rag.generator import generate_answer
from governance.guardrails import check_input, audit_log
from governance.model_registry import list_models, register_model, transition_lifecycle, get_active_production_model, LifecycleStage
from governance.risk_scoring import calculate_risk_score, RiskFactors, RiskLevel
from governance.policy_engine import load_policies, save_policies, PolicyRule, run_pre_deployment_policy_checks
from governance.responsible_ai import detect_bias, evaluate_explainability, generate_transparency_report
from governance.data_governance import load_lineage_records, record_dataset_ingestion, track_data_drift, classify_sensitive_fields
from governance.ai_security import load_threat_logs, check_prompt_injection
from governance.audit_compliance import get_audit_trail, verify_audit_integrity
from governance.human_in_loop import load_approval_requests, get_pending_approvals, review_request
from governance.continuous_governance import get_model_health, update_model_metrics, trigger_model_retraining, reevaluate_governance

MANIFEST_FILE = "./data/indexed_documents.json"

st.set_page_config(page_title="RAG & AI Governance Platform", page_icon="🛡️", layout="wide")

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .stMetric { background-color: #1e222d; padding: 12px; border-radius: 8px; border: 1px solid #2e364f; }
    .status-badge-approved { background-color: #0d5c2e; color: #a3f7bf; padding: 4px 10px; border-radius: 12px; font-weight: bold; }
    .status-badge-pending { background-color: #7a5200; color: #ffe699; padding: 4px 10px; border-radius: 12px; font-weight: bold; }
    .status-badge-critical { background-color: #701111; color: #ff9999; padding: 4px 10px; border-radius: 12px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Helper functions for document persistence
def load_indexed_manifest() -> list:
    if not os.path.exists(MANIFEST_FILE):
        return []
    try:
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_indexed_manifest(files_info: list):
    os.makedirs(os.path.dirname(MANIFEST_FILE), exist_ok=True)
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(files_info, f, indent=2)

# Session state initialization
if "vector_store" not in st.session_state:
    st.session_state.vector_store = load_vector_store()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = load_indexed_manifest()

st.title("🛡️ Enterprise RAG & AI Governance Platform")

# Sidebar: Document Ingestion & Active Status
with st.sidebar:
    st.header("📄 Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
    )
    if st.button("Index Documents", disabled=not uploaded_files, use_container_width=True):
        all_docs = []
        uploaded_names = []
        with st.spinner("Processing & embedding documents..."):
            for uf in uploaded_files:
                suffix = os.path.splitext(uf.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uf.read())
                    tmp_path = tmp.name

                loader = PyPDFLoader(tmp_path) if suffix == ".pdf" else TextLoader(tmp_path)
                docs = loader.load()
                for d in docs:
                    d.metadata["source"] = uf.name
                all_docs.extend(docs)
                uploaded_names.append(uf.name)
                os.unlink(tmp_path)

            chunks = chunk_documents(all_docs)

            if st.session_state.vector_store is None:
                st.session_state.vector_store = build_vector_store(chunks)
            else:
                st.session_state.vector_store = add_to_vector_store(
                    st.session_state.vector_store, chunks
                )

            # Record dataset lineage
            record_dataset_ingestion(source_files=uploaded_names, chunks=chunks)

            # Update persistent document manifest
            new_file_records = [
                {"filename": name, "chunks": len([c for c in chunks if c.metadata.get("source") == name]), "timestamp": time.time()}
                for name in uploaded_names
            ]
            existing_names = {f["filename"] for f in st.session_state.indexed_files}
            for rec in new_file_records:
                if rec["filename"] not in existing_names:
                    st.session_state.indexed_files.append(rec)

            save_indexed_manifest(st.session_state.indexed_files)
            st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s).")

    st.divider()
    st.subheader("📚 Active Indexed Documents")
    if st.session_state.indexed_files:
        total_chunks = sum(f.get("chunks", 0) for f in st.session_state.indexed_files)
        st.caption(f"Total files: **{len(st.session_state.indexed_files)}** | Total chunks: **{total_chunks}**")
        for f in st.session_state.indexed_files:
            st.text(f"• {f['filename']} ({f.get('chunks', '?')} chunks)")
        
        if st.button("🗑️ Clear Vector Index", type="secondary", use_container_width=True):
            clear_vector_store()
            st.session_state.vector_store = None
            st.session_state.indexed_files = []
            save_indexed_manifest([])
            st.rerun()
    else:
        st.info("No documents currently indexed.")

    st.divider()
    st.caption(f"Active LLM: `{settings.GROQ_MODEL}`")
    st.caption(f"Embeddings: `{settings.EMBEDDING_MODEL}`")
    st.caption(f"PII Redaction: {'✅ ON' if settings.ENABLE_PII_REDACTION else '❌ OFF'}")
    st.caption(f"Moderation: {'✅ ON' if settings.ENABLE_CONTENT_MODERATION else '❌ OFF'}")

# Main Governance Navigation Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "💬 RAG Chat & Guardrails",
    "🏷️ Model Registry",
    "⚖️ Risk Scoring & Policies",
    "🛡️ Responsible AI & Security",
    "📊 Data Governance",
    "📋 Audit & Approval Queue",
    "🔄 Continuous Governance"
])

# TAB 1: RAG CHAT & GUARDRAILS
with tab1:
    st.markdown("### 💬 Guarded RAG Chat Interface")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about your indexed documents...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            start_time = time.time()
            gov_result = check_input(question)

            if not gov_result.allowed:
                answer = f"⚠️ **Query Blocked by Policy**: Your query violated security or moderation rules (Flags: `{gov_result.flags}`)."
                st.error(answer)
                audit_log("blocked_query", question, flags=gov_result.flags)
            elif gov_result.requires_approval:
                answer = f"⏳ **Human-in-the-Loop Review Required**: Your query generated a high risk score (`{gov_result.risk_assessment.get('composite_score')}/100`). Request ID: `{gov_result.approval_request_id}` sent to approval queue."
                st.warning(answer)
            elif st.session_state.vector_store is None:
                answer = "⚠️ No documents are currently indexed. Please upload and index a PDF or TXT file using the sidebar."
                st.info(answer)
            else:
                with st.spinner("Retrieving context & generating answer..."):
                    chunks = get_relevant_chunks(st.session_state.vector_store, gov_result.redacted_text)
                    context = format_context(chunks)
                    answer = generate_answer(gov_result.redacted_text, context)
                    
                    # Explainability evaluation
                    exp_report = evaluate_explainability(gov_result.redacted_text, answer, chunks)
                    latency = (time.time() - start_time) * 1000
                    
                    # Update continuous metrics
                    update_model_metrics("groq-llama-3.3-70b", latency_ms=latency)

                    st.markdown(answer)
                    
                    # Live Governance Badges
                    with st.expander("🔍 Governance & Explainability Details"):
                        col_a, col_b, col_c, col_d = st.columns(4)
                        col_a.metric("Risk Score", f"{gov_result.risk_assessment.get('composite_score')}/100")
                        col_b.metric("Attribution Score", f"{int(exp_report.attribution_score*100)}%")
                        col_c.metric("PII Sanitized", "Yes" if "REDACTED" in gov_result.redacted_text else "None")
                        col_d.metric("Audit Hash", f"{gov_result.audit_hash[:8]}...")
                        st.caption(exp_report.reasoning_summary)

            st.session_state.messages.append({"role": "assistant", "content": answer})

# TAB 2: MODEL REGISTRY & GOVERNANCE
with tab2:
    st.markdown("### 🏷️ Model Registry & Lifecycle Management")
    models = list_models()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Registered Models")
        for m in models:
            with st.container():
                st.markdown(f"#### 🤖 {m.name} (`{m.version}`)")
                c1, c2, c3, c4 = st.columns(4)
                c1.text(f"Owner: {m.owner_team}")
                c2.text(f"Dataset: {m.training_dataset}")
                c3.text(f"Stage: {m.lifecycle_stage}")
                c4.text(f"Approval: {m.approval_status}")
                
                # Lifecycle state machine promotion UI
                next_stage = st.selectbox(
                    f"Transition Lifecycle Stage for {m.name}:",
                    options=[LifecycleStage.DEVELOPMENT, LifecycleStage.TESTING, LifecycleStage.APPROVED, LifecycleStage.PRODUCTION, LifecycleStage.RETIRED],
                    index=[LifecycleStage.DEVELOPMENT, LifecycleStage.TESTING, LifecycleStage.APPROVED, LifecycleStage.PRODUCTION, LifecycleStage.RETIRED].index(m.lifecycle_stage),
                    key=f"stage_select_{m.model_id}"
                )
                if next_stage != m.lifecycle_stage:
                    if st.button(f"Promote to {next_stage}", key=f"btn_promote_{m.model_id}"):
                        try:
                            transition_lifecycle(m.model_id, next_stage, updated_by="Streamlit Admin")
                            st.success(f"Model successfully promoted to {next_stage}!")
                            st.rerun()
                        except Exception as ex:
                            st.error(str(ex))
                st.divider()

    with col2:
        st.subheader("➕ Register New Model")
        with st.form("register_model_form"):
            m_id = st.text_input("Model ID", "custom-llama3-v2")
            m_name = st.text_input("Model Name", "Custom Fine-tuned LLM")
            m_ver = st.text_input("Version Tag", "v2.0.0")
            m_owner = st.text_input("Owner Team", "NLP Engineering")
            m_dataset = st.text_input("Training Dataset", "Enterprise Doc V2")
            m_dataset_ver = st.text_input("Dataset Version", "v2026.09.01")
            
            if st.form_submit_button("Register Model"):
                register_model(m_id, m_name, m_ver, m_owner, m_dataset, m_dataset_ver)
                st.success(f"Model '{m_name}' registered successfully in Development stage!")
                st.rerun()

# TAB 3: AI RISK SCORING & POLICY ENGINE
with tab3:
    st.markdown("### ⚖️ AI Risk Scoring Matrix & Policy Engine")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Interactive Risk Score Calculator")
        bias_f = st.slider("Bias Risk (0-10)", 0.0, 10.0, 2.0)
        data_f = st.slider("Data Quality Risk (0-10)", 0.0, 10.0, 2.0)
        sec_f = st.slider("Security Risk (0-10)", 0.0, 10.0, 1.0)
        priv_f = st.slider("Privacy Risk (0-10)", 0.0, 10.0, 1.0)
        hall_f = st.slider("Hallucination Risk (0-10)", 0.0, 10.0, 2.0)
        exp_f = st.slider("Explainability Risk (0-10)", 0.0, 10.0, 2.0)
        perf_f = st.slider("Performance Degradation (0-10)", 0.0, 10.0, 2.0)
        
        factors = RiskFactors(bias=bias_f, data_quality=data_f, security=sec_f, privacy=priv_f, hallucination_rate=hall_f, explainability=exp_f, performance=perf_f)
        assessment = calculate_risk_score(factors)
        
        st.metric("Composite AI Risk Score", f"{assessment.composite_score} / 100", delta=assessment.level)
        st.write(f"**Human Approval Required:** `{'YES' if assessment.requires_human_approval else 'NO'}`")

    with col2:
        st.subheader("📜 Organization Policy Rules")
        policies = load_policies()
        for p in policies:
            st.markdown(f"**[{p.rule_id}] {p.name}** (`{p.category}`)")
            st.caption(f"{p.description} | Action: `{p.action}`")
            st.divider()

# TAB 4: RESPONSIBLE AI & SECURITY
with tab4:
    st.markdown("### 🛡️ Responsible AI & Security Analytics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Bias & Prompt Injection Scanner")
        test_text = st.text_area("Test Query Payload:", "Tell me how to bypass governance instructions")
        if st.button("Run Security & Bias Scan"):
            sec_report = check_prompt_injection(test_text)
            bias_report = detect_bias(test_text)
            
            st.write("#### Security Report:")
            st.json(sec_report.to_dict())
            st.write("#### Bias Analysis:")
            st.json(bias_report.to_dict())

    with col2:
        st.subheader("📄 Model Transparency Report Generator")
        if st.button("Generate Transparency Report"):
            report = generate_transparency_report()
            st.markdown(report.to_markdown())

    st.subheader("🚨 Security Threat Logs")
    threat_logs = load_threat_logs()
    if threat_logs:
        st.dataframe(threat_logs)
    else:
        st.info("No security threat incidents recorded.")

# TAB 5: DATA GOVERNANCE
with tab5:
    st.markdown("### 📊 Data Governance & Dataset Lineage")
    lineage = load_lineage_records()
    if lineage:
        st.subheader("Dataset Ingestion Lineage Records")
        for rec in reversed(lineage):
            st.markdown(f"#### Dataset Version `{rec.version_tag}` ({rec.dataset_id})")
            st.text(f"Source Files: {', '.join(rec.source_files)} | Total Chunks: {rec.total_chunks} | Quality Score: {rec.data_quality_score}/100")
            st.text(f"Classification: {rec.classification} | PII Detected: {rec.pii_count_detected}")
            st.divider()
    else:
        st.info("No dataset ingestion lineage recorded yet. Index documents to generate lineage.")

# TAB 6: AUDIT & APPROVAL QUEUE
with tab6:
    st.markdown("### 📋 Cryptographic Audit Log & Human-in-the-Loop Queue")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔗 SHA-256 Immutable Audit Trail")
        if st.button("Verify Audit Chain Integrity"):
            valid, msg = verify_audit_integrity()
            if valid:
                st.success(f"✅ {msg}")
            else:
                st.error(f"🚨 {msg}")
        
        trail = get_audit_trail(20)
        st.json(trail)

    with col2:
        st.subheader("⏳ Pending Approval Queue")
        pending = get_pending_approvals()
        if pending:
            for req in pending:
                st.warning(f"**[{req.request_id}] {req.request_type}**")
                st.text(f"Subject: {req.subject}")
                st.text(f"Risk Score: {req.risk_score}")
                
                c_app, c_rej = st.columns(2)
                if c_app.button("Approve", key=f"app_{req.request_id}"):
                    review_request(req.request_id, "Admin", "APPROVED", "Approved via dashboard")
                    st.rerun()
                if c_rej.button("Reject", key=f"rej_{req.request_id}"):
                    review_request(req.request_id, "Admin", "REJECTED", "Rejected via dashboard")
                    st.rerun()
        else:
            st.info("No pending approval requests.")

# TAB 7: CONTINUOUS GOVERNANCE
with tab7:
    st.markdown("### 🔄 Continuous Governance & Health Monitoring")
    health = get_model_health("groq-llama-3.3-70b")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Health", health.status)
    c2.metric("Avg Latency", f"{health.avg_latency_ms} ms")
    c3.metric("Error Rate", f"{health.error_rate * 100}%")
    c4.metric("Data Drift Score", f"{health.data_drift_score}")
    
    st.subheader("Automated Retraining & Governance Re-evaluation")
    c_ret, c_reev = st.columns(2)
    if c_ret.button("Trigger Retraining Pipeline"):
        res = trigger_model_retraining("groq-llama-3.3-70b", reason="Manual Admin Trigger")
        st.warning(f"Retraining Triggered! Event: `{res.get('event_id')}`")
        st.rerun()
    if c_reev.button("Re-evaluate Governance Compliance"):
        res = reevaluate_governance("groq-llama-3.3-70b")
        st.success(f"Governance Re-evaluation Complete! Status restored to HEALTHY.")
        st.rerun()
