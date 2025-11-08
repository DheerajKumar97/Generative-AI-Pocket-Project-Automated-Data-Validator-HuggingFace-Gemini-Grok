import streamlit as st
import pandas as pd
import json
from datetime import datetime
import io

# Import backend classes
from main import FileHandler, MultiModelLLMClient, ComprehensiveRAGValidator

# Page configuration
st.set_page_config(
    page_title="Multi-Model RAG Data Validator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(120deg, #1f77b4, #ff7f0e, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .model-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: bold;
        margin: 0.25rem;
    }
    .huggingface-badge {
        background-color: #FFD21E;
        color: #000;
    }
    .gemini-badge {
        background-color: #4285f4;
        color: #fff;
    }
    .grok-badge {
        background-color: #000;
        color: #fff;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        color: #155724;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        color: #856404;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        color: #721c24;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar with configuration options"""
    with st.sidebar:
        st.header("⚙️ AI Model Configuration")
        
        # Model Provider Selection
        provider = st.selectbox(
            "🤖 AI Provider",
            ["HuggingFace", "Gemini", "Grok"],
            help="Select the AI provider for validation analysis"
        )
        
        # Display provider badge and get model options
        if provider == "HuggingFace":
            st.markdown('<span class="model-badge huggingface-badge">🤗 HuggingFace</span>', unsafe_allow_html=True)
            model_options = [
                "Qwen/Qwen2.5-72B-Instruct",
                "meta-llama/Llama-2-70b-chat-hf",
                "mistralai/Mixtral-8x7B-Instruct-v0.1"
            ]
        elif provider == "Gemini":
            st.markdown('<span class="model-badge gemini-badge">✨ Gemini</span>', unsafe_allow_html=True)
            model_options = [
                "gemini-2.0-flash-exp",
                "gemini-1.5-pro",
                "gemini-1.5-flash"
            ]
        else:  # Grok
            st.markdown('<span class="model-badge grok-badge">🚀 Grok (xAI)</span>', unsafe_allow_html=True)
            model_options = [
                "llama-3.3-70b-versatile",
                "llama-3.1-70b-versatile",
                "mixtral-8x7b-32768"
            ]
        
        # Model Selection
        model_name = st.selectbox(
            "📦 Model",
            model_options,
            help=f"Select the {provider} model for analysis"
        )
        
        # API Token
        api_key = st.text_input(
            f"🔑 {provider} API Key",
            type="password",
            help=f"Enter your {provider} API key"
        )
        
        st.divider()
        
        # File Upload
        st.header("📁 Upload Data")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls', 'json', 'txt'],
            help="Supported formats: CSV, Excel, JSON, TXT"
        )
        
        # Encoding Selection
        encoding_option = st.selectbox(
            "📝 Encoding (optional)",
            ["Auto-detect", "utf-8", "latin-1", "iso-8859-1", "cp1252", "utf-16"],
            help="Select encoding or use auto-detection"
        )
        
        st.divider()
        
        # Validation Selection
        st.header("✅ Validations")
        validations = [
            "Data Type Check",
            "Range Check",
            "Null Value Check",
            "Duplicate Detection",
            "Outlier Detection",
            "Data Integrity Check",
            "Quality Scoring"
        ]
        
        selected_validations = st.multiselect(
            "Select validations to run",
            validations,
            default=validations,
            help="Choose which validations to perform"
        )
        
        st.divider()
        
        # Domain Rules
        st.header("📋 Domain Rules")
        with st.expander("Add Custom Rules"):
            rule_key = st.text_input("Rule Name")
            rule_value = st.text_area("Rule Description")
            
            if 'domain_rules' not in st.session_state:
                st.session_state.domain_rules = {}
            
            if st.button("Add Rule") and rule_key and rule_value:
                st.session_state.domain_rules[rule_key] = rule_value
                st.success(f"Added rule: {rule_key}")
            
            # Display existing rules
            if st.session_state.domain_rules:
                st.write("**Current Rules:**")
                for key, value in st.session_state.domain_rules.items():
                    st.text(f"• {key}: {value[:50]}...")
    
    return provider, model_name, api_key, uploaded_file, encoding_option, selected_validations


def render_file_info(uploaded_file, provider):
    """Render file information section"""
    st.header("📊 File Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("File Name", uploaded_file.name)
    with col2:
        file_size = uploaded_file.size / 1024
        st.metric("File Size", f"{file_size:.2f} KB")
    with col3:
        file_type = uploaded_file.name.split('.')[-1].upper()
        st.metric("File Type", file_type)
    with col4:
        st.metric("AI Provider", provider)


def load_dataframe(uploaded_file, encoding_option):
    """Load dataframe from uploaded file"""
    file_handler = FileHandler()
    encoding = None if encoding_option == "Auto-detect" else encoding_option
    
    if uploaded_file.name.endswith('.csv'):
        return file_handler.read_csv(uploaded_file, encoding)
    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
        return file_handler.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.json'):
        return file_handler.read_json(uploaded_file, encoding)
    elif uploaded_file.name.endswith('.txt'):
        return file_handler.read_txt(uploaded_file, encoding)
    else:
        raise ValueError("Unsupported file format")


def render_data_preview(df):
    """Render data preview section"""
    st.header("👀 Data Preview")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.dataframe(df.head(10), use_container_width=True)
    with col2:
        st.write("**Column Types:**")
        for col, dtype in df.dtypes.items():
            st.text(f"{col}: {dtype}")


def render_data_summary(df):
    """Render data summary metrics"""
    st.header("📈 Data Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        null_count = df.isnull().sum().sum()
        null_pct = (null_count / (len(df) * len(df.columns)) * 100) if len(df) > 0 else 0
        st.metric("Null Values", f"{null_count:,}", f"{null_pct:.1f}%")
    with col4:
        duplicates = df.duplicated().sum()
        dup_pct = (duplicates / len(df) * 100) if len(df) > 0 else 0
        st.metric("Duplicates", f"{duplicates:,}", f"{dup_pct:.1f}%")
    with col5:
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Memory", f"{memory_mb:.2f} MB")


def render_validation_results(results, provider, model_name):
    """Render validation results section"""
    st.header("🔍 Validation Results")
    
    for validation_name, result in results.items():
        with st.expander(f"📋 {validation_name}", expanded=True):
            if "error" in result:
                st.error(f"❌ Error: {result['error']}")
            else:
                # Show which AI model analyzed this
                if provider == "HuggingFace":
                    st.markdown(f'<span class="model-badge huggingface-badge">Analyzed by 🤗 {model_name}</span>', unsafe_allow_html=True)
                elif provider == "Gemini":
                    st.markdown(f'<span class="model-badge gemini-badge">Analyzed by ✨ {model_name}</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="model-badge grok-badge">Analyzed by 🚀 {model_name}</span>', unsafe_allow_html=True)
                
                # Findings
                st.subheader("📊 Findings")
                findings = result.get('findings', {})
                
                if 'pass' in findings:
                    status = "✅ PASS" if findings['pass'] else "❌ FAIL"
                    st.markdown(f"**Status:** {status}")
                
                # Show key metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'total_nulls' in findings:
                        st.metric("Total Nulls", findings['total_nulls'])
                    if 'total_duplicates' in findings:
                        st.metric("Total Duplicates", findings['total_duplicates'])
                    if 'total_outliers' in findings:
                        st.metric("Total Outliers", findings['total_outliers'])
                
                with col2:
                    if 'duplicate_percentage' in findings:
                        st.metric("Duplicate %", f"{findings['duplicate_percentage']:.2f}%")
                    if 'quality_scores' in findings:
                        overall = findings['quality_scores'].get('overall_quality_score', 0)
                        st.metric("Quality Score", f"{overall:.1f}/100")
                
                # Detailed findings
                with st.expander("📄 Detailed Findings"):
                    st.json(findings)
                
                # AI Analysis
                st.subheader("🤖 AI Analysis")
                analysis = result.get('ai_analysis', '')
                st.markdown(analysis)


def render_quality_assessment(validator, results):
    """Render overall quality assessment"""
    st.header("🎯 Overall Quality Assessment")
    quality_score = validator.quality_score
    
    # Determine quality level and color
    if quality_score >= 95:
        quality_level = "Perfect"
        quality_color = "🟢"
        quality_message = "Outstanding! Production-ready data."
    elif quality_score >= 90:
        quality_level = "Excellent"
        quality_color = "🟢"
        quality_message = "High quality data with minimal issues."
    elif quality_score >= 80:
        quality_level = "Good"
        quality_color = "🟡"
        quality_message = "Acceptable quality, minor improvements needed."
    elif quality_score >= 70:
        quality_level = "Fair"
        quality_color = "🟠"
        quality_message = "Moderate issues found, attention required."
    elif quality_score >= 60:
        quality_level = "Poor"
        quality_color = "🔴"
        quality_message = "Significant issues, not production-ready."
    else:
        quality_level = "Critical"
        quality_color = "🔴"
        quality_message = "Critical data quality issues detected!"
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.metric(
            "Overall Quality Score",
            f"{quality_color} {quality_score:.1f}/100"
        )
        st.progress(quality_score / 100)
        
        if quality_score >= 90:
            st.success(f"✅ {quality_level}: {quality_message}")
        elif quality_score >= 80:
            st.info(f"ℹ️ {quality_level}: {quality_message}")
        elif quality_score >= 70:
            st.warning(f"⚠️ {quality_level}: {quality_message}")
        else:
            st.error(f"❌ {quality_level}: {quality_message}")
    
    # Show penalty breakdown
    if "Quality Scoring" in results and quality_score < 100:
        with st.expander("📉 Quality Score Breakdown", expanded=(quality_score < 80)):
            findings = results["Quality Scoring"]["findings"]
            
            if "quality_scores" in findings:
                penalty_details = findings["quality_scores"].get("penalty_details", [])
                issues_found = findings["quality_scores"].get("issues_found", 0)
                
                st.write(f"**Total Issues Found:** {issues_found}")
                st.write(f"**Points Deducted:** {100 - quality_score:.1f}")
                
                if penalty_details:
                    st.write("\n**Detailed Penalties:**")
                    for penalty in penalty_details:
                        st.text(f"• {penalty}")
                
                # Show specific issue counts
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    null_pct = findings.get("null_percentage", 0)
                    st.metric("Missing Data", f"{null_pct:.2f}%", 
                             delta=f"-{null_pct * 10:.1f} pts" if null_pct > 0 else "No issues")
                
                with col2:
                    dup_pct = findings.get("duplicate_percentage", 0)
                    st.metric("Duplicates", f"{dup_pct:.2f}%",
                             delta=f"-{dup_pct * 15:.1f} pts" if dup_pct > 0 else "No issues")
                
                with col3:
                    validity_issues = findings.get("validity_issues", 0)
                    st.metric("Validity Issues", validity_issues,
                             delta=f"-{validity_issues * 10:.1f} pts" if validity_issues > 0 else "No issues")


def render_quality_breakdown(results):
    """Render quality breakdown section"""
    st.subheader("📊 Quality Breakdown")
    if "Quality Scoring" in results and "findings" in results["Quality Scoring"]:
        scores = results["Quality Scoring"]["findings"].get("quality_scores", {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            completeness = scores.get('completeness', 0)
            st.metric("Completeness", f"{completeness:.1f}%")
            st.progress(completeness / 100)
        
        with col2:
            uniqueness = scores.get('uniqueness', 0)
            st.metric("Uniqueness", f"{uniqueness:.1f}%")
            st.progress(uniqueness / 100)
        
        with col3:
            validity = scores.get('validity', 0)
            st.metric("Validity", f"{validity:.1f}%")
            st.progress(validity / 100)


def render_recommendations(results, quality_score):
    """Render recommendations section"""
    st.header("💡 Key Recommendations")
    
    recommendations = []
    critical_count = 0
    high_count = 0
    medium_count = 0
    
    for validation_name, result in results.items():
        if "error" not in result:
            findings = result.get('findings', {})
            
            if 'total_nulls' in findings and findings['total_nulls'] > 0:
                pct = findings.get('null_percentage', 0)
                if pct > 5:
                    critical_count += 1
                    recommendations.append(f"🔴 **CRITICAL**: {validation_name} - {findings['total_nulls']} missing values ({pct:.1f}%)")
                else:
                    high_count += 1
                    recommendations.append(f"🟠 **HIGH**: {validation_name} - {findings['total_nulls']} missing values")
            
            if 'total_duplicates' in findings and findings['total_duplicates'] > 0:
                pct = findings.get('duplicate_percentage', 0)
                if pct > 1:
                    critical_count += 1
                    recommendations.append(f"🔴 **CRITICAL**: {validation_name} - {findings['total_duplicates']} duplicates ({pct:.1f}%)")
                else:
                    high_count += 1
                    recommendations.append(f"🟠 **HIGH**: {validation_name} - {findings['total_duplicates']} duplicates")
            
            if 'critical_issues' in findings and findings['critical_issues'] > 0:
                critical_count += findings['critical_issues']
                recommendations.append(f"🔴 **CRITICAL**: {validation_name} - {findings['critical_issues']} integrity violations")
            
            if 'total_extreme_outliers' in findings and findings['total_extreme_outliers'] > 0:
                high_count += 1
                recommendations.append(f"🟠 **HIGH**: {validation_name} - {findings['total_extreme_outliers']} extreme outliers detected")
            
            if 'violations' in findings and findings['violations']:
                medium_count += len(findings['violations'])
                for field, info in list(findings['violations'].items())[:3]:
                    recommendations.append(f"🟡 **MEDIUM**: {validation_name} - {field} has range violations")
    
    if critical_count > 0 or high_count > 0:
        st.error(f"⚠️ **Found {critical_count} CRITICAL and {high_count} HIGH priority issues**")
    elif medium_count > 0:
        st.warning(f"⚠️ **Found {medium_count} MEDIUM priority issues**")
    
    if recommendations:
        st.markdown("### Priority Actions:")
        for rec in recommendations[:15]:
            st.markdown(rec)
    else:
        st.success("✅ No critical issues found! Your data quality is excellent.")
    
    # Show improvement suggestions
    if quality_score < 95:
        st.markdown("### 🎯 To Achieve Production-Ready Quality (95+):")
        
        improvement_tips = []
        
        if "Quality Scoring" in results:
            qf = results["Quality Scoring"]["findings"]
            
            if qf.get("null_percentage", 0) > 0:
                improvement_tips.append("• **Eliminate all missing values** - Use imputation or collect complete data")
            
            if qf.get("duplicate_percentage", 0) > 0:
                improvement_tips.append("• **Remove all duplicate records** - Ensure unique identifiers")
            
            if qf.get("validity_issues", 0) > 0:
                improvement_tips.append("• **Fix validity issues** - Correct negative values, empty strings, and invalid data")
            
            if qf.get("consistency_issues", 0) > 0:
                improvement_tips.append("• **Resolve consistency issues** - Ensure uniform data types and formats")
        
        if "Outlier Detection" in results and results["Outlier Detection"]["findings"].get("total_extreme_outliers", 0) > 0:
            improvement_tips.append("• **Address extreme outliers** - Investigate and handle anomalous values")
        
        if improvement_tips:
            for tip in improvement_tips:
                st.markdown(tip)
        else:
            st.markdown("• Continue monitoring data quality metrics")
            st.markdown("• Implement automated validation in your data pipeline")


def render_download_reports(uploaded_file, provider, model_name, results, quality_score, df):
    """Render download reports section"""
    st.header("📥 Download Reports")
    
    # Prepare report data
    report_data = {
        "metadata": {
            "file_name": uploaded_file.name,
            "validation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ai_provider": provider,
            "ai_model": model_name,
            "rows": len(df),
            "columns": len(df.columns),
            "quality_score": quality_score
        },
        "validations": results
    }
    
    report_json = json.dumps(report_data, indent=2, default=str)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            "📄 Download JSON Report",
            report_json,
            file_name=f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # Create text report
        text_report = f"""
{'='*80}
DATA VALIDATION REPORT
{'='*80}

File: {uploaded_file.name}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
AI Provider: {provider}
AI Model: {model_name}
Quality Score: {quality_score:.2f}/100

Dataset Overview:
- Rows: {len(df):,}
- Columns: {len(df.columns)}
- Null Values: {df.isnull().sum().sum():,}
- Duplicates: {df.duplicated().sum():,}

{'='*80}
VALIDATION RESULTS
{'='*80}

"""
        for validation_name, result in results.items():
            text_report += f"\n{validation_name}\n{'-'*80}\n"
            if "error" in result:
                text_report += f"ERROR: {result['error']}\n"
            else:
                text_report += f"\nFindings:\n{json.dumps(result.get('findings', {}), indent=2, default=str)}\n"
                text_report += f"\nAI Analysis:\n{result.get('ai_analysis', '')}\n"
            text_report += "\n"
        
        text_report += f"\n{'='*80}\nEND OF REPORT\n{'='*80}\n"
        
        st.download_button(
            "📝 Download Text Report",
            text_report,
            file_name=f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col3:
        # Create CSV summary
        summary_data = []
        for validation_name, result in results.items():
            if "error" not in result:
                findings = result.get('findings', {})
                summary_data.append({
                    "Validation": validation_name,
                    "Status": "PASS" if findings.get('pass', False) else "FAIL",
                    "Key_Metric": json.dumps(findings, default=str)[:100]
                })
        
        summary_df = pd.DataFrame(summary_data)
        csv_buffer = io.StringIO()
        summary_df.to_csv(csv_buffer, index=False)
        
        st.download_button(
            "📊 Download CSV Summary",
            csv_buffer.getvalue(),
            file_name=f"validation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )


def render_landing_page():
    """Render landing page when no file is uploaded"""
    st.info("👆 Upload a file using the sidebar to get started")
    
    # Features Grid
    st.header("🌟 Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📁 Multi-Format Support
        - **CSV** with auto-encoding detection
        - **Excel** (XLSX, XLS)
        - **JSON** (objects and arrays)
        - **TXT** (delimited files)
        
        ### 🤖 Multi-Model AI
        - **HuggingFace** (Qwen, Llama, Mixtral)
        - **Google Gemini** (2.0 Flash, 1.5 Pro)
        - **Grok/xAI** (via Groq API)
        """)
    
    with col2:
        st.markdown("""
        ### 🔍 Comprehensive Validations
        - Data Type Analysis
        - Range & Outlier Detection
        - Null Value Identification
        - Duplicate Detection
        - Statistical Analysis
        - Quality Scoring
        
        ### 📊 Smart Reports
        - JSON detailed reports
        - Text summaries
        - CSV exports
        - Quality metrics dashboard
        """)
    
    # Model Comparison
    st.header("🤖 AI Model Comparison")
    
    comparison_data = {
        "Provider": ["HuggingFace", "Google Gemini", "Grok (xAI)"],
        "Best For": [
            "Open-source models, customization",
            "Fast responses, multimodal",
            "Conversational analysis, speed"
        ],
        "Response Time": ["Medium", "Fast", "Very Fast"],
        "Cost": ["Pay per token", "Free tier available", "Pay per token"],
        "Strengths": [
            "Model variety, fine-tuning",
            "Latest Google AI, integration",
            "Real-time analysis, accuracy"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Quick Start Guide
    st.header("🚀 Quick Start Guide")
    
    tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ Setup", "2️⃣ Upload", "3️⃣ Configure", "4️⃣ Validate"])
    
    with tab1:
        st.markdown("""
        ### Get Your API Keys
        
        **HuggingFace:**
        1. Go to [huggingface.co](https://huggingface.co)
        2. Sign up/Login
        3. Go to Settings → Access Tokens
        4. Create new token with "Read" permission
        
        **Google Gemini:**
        1. Go to [ai.google.dev](https://ai.google.dev)
        2. Get API key from Google AI Studio
        3. Copy your API key
        
        **Grok (via Groq):**
        1. Go to [console.groq.com](https://console.groq.com)
        2. Sign up for Groq Cloud
        3. Create API key in dashboard
        """)
    
    with tab2:
        st.markdown("""
        ### Upload Your Data
        
        1. Click **"Browse files"** in the sidebar
        2. Select your data file (CSV, Excel, JSON, or TXT)
        3. Choose encoding if needed (auto-detect usually works)
        4. Wait for file to load
        
        **Encoding Issues?**
        - Try "Auto-detect" first
        - If that fails, try "latin-1" or "utf-8"
        - Check file encoding in your text editor
        """)
    
    with tab3:
        st.markdown("""
        ### Configure Validation
        
        1. **Select AI Provider**: Choose HuggingFace, Gemini, or Grok
        2. **Pick a Model**: Select specific model for analysis
        3. **Enter API Key**: Paste your API key
        4. **Choose Validations**: Select which checks to run
        5. **Add Rules** (optional): Define custom domain rules
        
        **Recommended Validations:**
        - Data Type Check
        - Null Value Check
        - Quality Scoring
        """)
    
    with tab4:
        st.markdown("""
        ### Run Validation
        
        1. Click **"🚀 Run Validation"** button
        2. Wait for analysis (30 seconds - 2 minutes)
        3. Review results in expandable sections
        4. Check overall quality score
        5. Download reports (JSON, TXT, or CSV)
        
        **Interpreting Results:**
        - 🟢 95-100: Perfect - Production-ready
        - 🟢 90-94: Excellent - High quality
        - 🟡 80-89: Good - Minor improvements needed
        - 🟠 70-79: Fair - Attention required
        - 🔴 60-69: Poor - Not production-ready
        - 🔴 <60: Critical - Major problems
        """)
    
    # FAQ Section
    st.header("❓ Frequently Asked Questions")
    
    with st.expander("Which AI model should I choose?"):
        st.markdown("""
        - **HuggingFace (Qwen)**: Best for detailed, technical analysis
        - **Gemini 2.0 Flash**: Fastest responses, good for quick checks
        - **Grok**: Balanced speed and accuracy, conversational insights
        
        Try different models to compare results!
        """)
    
    with st.expander("How do I handle encoding errors?"):
        st.markdown("""
        1. Try "Auto-detect" first
        2. If file fails, select "latin-1" or "utf-8"
        3. Check your file's actual encoding in a text editor
        4. Common encodings: UTF-8 (modern), Latin-1 (European), CP1252 (Windows)
        """)
    
    with st.expander("What if validation takes too long?"):
        st.markdown("""
        - Large files (>10MB) may take 2-5 minutes
        - Reduce number of selected validations
        - Try Gemini Flash for fastest results
        - Check your internet connection
        - Ensure API key is valid
        """)
    
    with st.expander("Can I validate multiple files?"):
        st.markdown("""
        Currently, you can validate one file at a time. To validate multiple files:
        1. Upload and validate first file
        2. Download reports
        3. Upload next file
        4. Repeat process
        
        Each file gets independent analysis and reports.
        """)

    st.markdown("<h3 style='text-align: center;'>🤝 Connect With Me</h3>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="display: flex; justify-content: center; align-items: center; gap: 30px; margin-top: 10px;">
            <a href="https://www.linkedin.com/in/dheerajkumar1997/" target="_blank" style="text-decoration: none;">
                <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="40" height="40" alt="LinkedIn">
            </a>
            <a href="https://github.com/DheerajKumar97?tab=repositories" target="_blank" style="text-decoration: none;">
                <img src="https://cdn-icons-png.flaticon.com/512/733/733553.png" width="40" height="40" alt="GitHub">
            </a>
            <a href="https://dheeraj-kumar-k.lovable.app/" target="_blank" style="text-decoration: none;">
                <img src="https://cdn-icons-png.flaticon.com/512/841/841364.png" width="40" height="40" alt="Website">
            </a>
        </div>
        <br>
        <br>
        <p style="text-align: center; font-size: 14px; color: gray;">
            Made By Dheeraj Kumar K<br> Copyright © 2025 All rights reserved 
        </p>
        """,
        unsafe_allow_html=True
    )


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">🔍 LLM Powered RAG Based Multi-Model Data Validator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Data Quality Validation with 24 Techniques Using HuggingFace, Gemini & Grok</div>', unsafe_allow_html=True)
    
    # Render sidebar and get configurations
    provider, model_name, api_key, uploaded_file, encoding_option, selected_validations = render_sidebar()
    
    # Main Content
    if uploaded_file is not None:
        try:
            # File Information
            render_file_info(uploaded_file, provider)
            
            # Load Data
            with st.spinner("🔄 Loading data..."):
                df = load_dataframe(uploaded_file, encoding_option)
            
            st.success(f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            
            # Data Preview
            render_data_preview(df)
            
            # Data Summary
            render_data_summary(df)
            
            # Run Validation
            if st.button("🚀 Run Validation", type="primary", use_container_width=True):
                if not api_key:
                    st.error(f"⚠️ Please enter your {provider} API key in the sidebar")
                    return
                
                if not selected_validations:
                    st.warning("⚠️ Please select at least one validation")
                    return
                
                try:
                    # Initialize Multi-Model LLM Client
                    with st.spinner(f"🔧 Initializing {provider} client..."):
                        llm_client = MultiModelLLMClient(
                            provider=provider.lower(),
                            api_key=api_key,
                            model_name=model_name
                        )
                    
                    # Initialize Validator
                    with st.spinner("🔧 Initializing validator..."):
                        validator = ComprehensiveRAGValidator(llm_client)
                    
                    # Build Knowledge Base
                    with st.spinner("📚 Building knowledge base..."):
                        domain_rules = st.session_state.get('domain_rules', {})
                        if not domain_rules:
                            domain_rules = {
                                "data_quality": "Ensure data completeness and accuracy",
                                "consistency": "Check for logical consistency across fields",
                                "integrity": "Maintain referential integrity and valid relationships"
                            }
                        validator.build_knowledge_base(df, domain_rules)
                    
                    st.success(f"✅ Using {provider} with model: {model_name}")
                    
                    # Run Validations
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(value):
                        progress_bar.progress(value)
                    
                    def update_status(text):
                        status_text.text(text)
                    
                    results = validator.run_selected_validations(
                        df, 
                        selected_validations,
                        progress_callback=update_progress,
                        status_callback=update_status
                    )
                    
                    # Display Results
                    render_validation_results(results, provider, model_name)
                    
                    # Overall Quality Assessment
                    render_quality_assessment(validator, results)
                    
                    # Quality Breakdown
                    render_quality_breakdown(results)
                    
                    # Recommendations
                    render_recommendations(results, validator.quality_score)
                    
                    # Download Reports
                    render_download_reports(uploaded_file, provider, model_name, results, validator.quality_score, df)
                    
                    # Model Comparison Suggestion
                    if validator.quality_score < 80:
                        st.info(f"💡 **Tip:** Try running validation with a different AI model to get alternative insights! Currently using {provider}.")
                    
                except Exception as e:
                    st.error(f"❌ Validation Error: {str(e)}")
                    with st.expander("🔍 Error Details"):
                        st.exception(e)
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.info("💡 Try selecting a different encoding from the sidebar")
            with st.expander("🔍 Error Details"):
                st.exception(e)
    
    else:
        # Landing Page
        render_landing_page()


if __name__ == "__main__":
    main()