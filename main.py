import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json
from datetime import datetime
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import faiss
from scipy import stats
from fuzzywuzzy import fuzz
import re
import io
import chardet
import openpyxl
from pathlib import Path
from groq import Groq
import google.generativeai as genai


class FileHandler:
    """Handle various file formats with encoding detection"""
    
    @staticmethod
    def detect_encoding(file_bytes: bytes) -> str:
        """Detect file encoding"""
        result = chardet.detect(file_bytes)
        return result['encoding'] or 'utf-8'
    
    @staticmethod
    def read_csv(file, encoding: str = None) -> pd.DataFrame:
        """Read CSV with automatic encoding detection"""
        try:
            if encoding:
                return pd.read_csv(file, encoding=encoding)
            
            # Try UTF-8 first
            try:
                file.seek(0)
                return pd.read_csv(file, encoding='utf-8')
            except UnicodeDecodeError:
                pass
            
            # Try detecting encoding
            file.seek(0)
            raw_data = file.read()
            detected_encoding = FileHandler.detect_encoding(raw_data)
            file.seek(0)
            
            try:
                return pd.read_csv(io.BytesIO(raw_data), encoding=detected_encoding)
            except:
                # Try common encodings
                for enc in ['latin-1', 'iso-8859-1', 'cp1252', 'utf-16']:
                    try:
                        file.seek(0)
                        return pd.read_csv(file, encoding=enc)
                    except:
                        continue
                
                raise ValueError("Could not decode CSV file with any known encoding")
                
        except Exception as e:
            raise ValueError(f"Error reading CSV: {str(e)}")
    
    @staticmethod
    def read_excel(file) -> pd.DataFrame:
        """Read Excel file"""
        try:
            return pd.read_excel(file, engine='openpyxl')
        except Exception as e:
            raise ValueError(f"Error reading Excel: {str(e)}")
    
    @staticmethod
    def read_json(file, encoding: str = None) -> pd.DataFrame:
        """Read JSON with encoding detection"""
        try:
            file.seek(0)
            raw_data = file.read()
            
            if encoding is None:
                detected_encoding = FileHandler.detect_encoding(raw_data)
            else:
                detected_encoding = encoding
            
            json_str = raw_data.decode(detected_encoding)
            data = json.loads(json_str)
            
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                return pd.DataFrame([data])
            else:
                raise ValueError("JSON must be an object or array")
                
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error reading JSON: {str(e)}")
    
    @staticmethod
    def read_txt(file, encoding: str = None) -> pd.DataFrame:
        """Read TXT file (assumes CSV-like format)"""
        try:
            file.seek(0)
            raw_data = file.read()
            
            if encoding is None:
                detected_encoding = FileHandler.detect_encoding(raw_data)
            else:
                detected_encoding = encoding
            
            # Try to detect delimiter
            text_sample = raw_data.decode(detected_encoding, errors='ignore')[:1000]
            
            delimiters = [',', '\t', '|', ';']
            delimiter = ','
            max_cols = 0
            
            for delim in delimiters:
                cols = len(text_sample.split('\n')[0].split(delim))
                if cols > max_cols:
                    max_cols = cols
                    delimiter = delim
            
            file.seek(0)
            return pd.read_csv(
                io.BytesIO(raw_data),
                encoding=detected_encoding,
                delimiter=delimiter,
                on_bad_lines='skip'
            )
            
        except Exception as e:
            raise ValueError(f"Error reading TXT: {str(e)}")


class MultiModelLLMClient:
    """Unified client for multiple LLM providers"""
    
    def __init__(self, provider: str, api_key: str, model_name: str = None):
        self.provider = provider.lower()
        self.api_key = api_key
        self.model_name = model_name
        self.client = None
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate client based on provider"""
        if self.provider == "huggingface":
            self.client = InferenceClient(token=self.api_key)
            if not self.model_name:
                self.model_name = "Qwen/Qwen2.5-72B-Instruct"
        
        elif self.provider == "gemini":
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(
                self.model_name or "gemini-2.0-flash-exp"
            )
        
        elif self.provider == "grok":
            self.client = Groq(api_key=self.api_key)
            if not self.model_name:
                self.model_name = "llama-3.3-70b-versatile"
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def query(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.2) -> str:
        """Query the LLM with unified interface"""
        try:
            if self.provider == "huggingface":
                return self._query_huggingface(prompt, max_tokens, temperature)
            elif self.provider == "gemini":
                return self._query_gemini(prompt, max_tokens, temperature)
            elif self.provider == "grok":
                return self._query_grok(prompt, max_tokens, temperature)
        except Exception as e:
            return f"⚠️ **{self.provider.upper()} Error**: {str(e)}"
    
    def _query_huggingface(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Query HuggingFace API"""
        messages = [{"role": "user", "content": prompt}]
        response = ""
        
        for message in self.client.chat_completion(
            messages=messages,
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True
        ):
            response += message.choices[0].delta.content or ""
        
        return response
    
    def _query_gemini(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Query Gemini API"""
        generation_config = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        
        response = self.client.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        return response.text
    
    def _query_grok(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Query Grok (via Groq) API"""
        messages = [{"role": "user", "content": prompt}]
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content


class ComprehensiveRAGValidator:
    """Multi-Model RAG-based Data Validation System"""
    
    def __init__(self, llm_client: MultiModelLLMClient):
        self.llm_client = llm_client
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge_base = []
        self.index = None
        self.validation_results = {}
        self.quality_score = 0
        
    def build_knowledge_base(self, df: pd.DataFrame, domain_rules: Dict[str, Any]):
        """Build comprehensive knowledge base for RAG"""
        kb = []
        
        # Schema information
        for col in df.columns:
            kb.append(f"Column '{col}': dtype={df[col].dtype}, nulls={df[col].isnull().sum()}, unique={df[col].nunique()}")
            if df[col].dtype in ['int64', 'float64']:
                try:
                    stats_info = f"'{col}' stats: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}, std={df[col].std():.2f}"
                    kb.append(stats_info)
                except:
                    pass
        
        # Domain rules
        for rule_type, rule_content in domain_rules.items():
            kb.append(f"Rule: {rule_type} - {rule_content}")
        
        self.knowledge_base = kb
        embeddings = self.embedding_model.encode(kb)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings.astype('float32'))
        
    def retrieve_context(self, query: str, k: int = 5) -> str:
        """Retrieve relevant context using semantic search"""
        if not self.knowledge_base or self.index is None:
            return "No context available"
        
        query_embedding = self.embedding_model.encode([query])
        _, indices = self.index.search(query_embedding.astype('float32'), k)
        return "\n".join([self.knowledge_base[i] for i in indices[0]])
    
    def create_validation_prompt(self, validation_name: str, context: str, 
                                findings: Dict[str, Any]) -> str:
        """Create structured prompt for LLM validation"""
        return f"""You are a data quality expert. Analyze this validation result concisely:

VALIDATION: {validation_name}
CONTEXT: {context}
FINDINGS: {json.dumps(findings, indent=2, default=str)}

Provide a brief analysis with:
1. Status: PASS/FAIL/WARNING
2. Severity: CRITICAL/HIGH/MEDIUM/LOW
3. Key Issues (if any)
4. Quick Recommendation

Keep response under 300 words."""
    
    # ================== VALIDATION METHODS ==================
    
    def validate_1_data_type_check(self, df: pd.DataFrame) -> Dict[str, Any]:
        """1. Data Type Check"""
        type_summary = {}
        for col in df.columns:
            type_summary[col] = {
                "dtype": str(df[col].dtype),
                "sample_values": str(df[col].dropna().head(3).tolist())[:100]
            }
        
        findings = {
            "type_summary": type_summary,
            "total_columns": len(df.columns)
        }
        
        context = self.retrieve_context("data type validation")
        prompt = self.create_validation_prompt("Data Type Check", context, findings)
        analysis = self.llm_client.query(prompt)
        
        return {"findings": findings, "ai_analysis": analysis}
    
    def validate_2_range_check(self, df: pd.DataFrame) -> Dict[str, Any]:
        """2. Range Check for numeric columns"""
        violations = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:10]:
            col_min = df[col].min()
            col_max = df[col].max()
            col_mean = df[col].mean()
            col_std = df[col].std()
            
            if col_std > 0:
                outliers_lower = (df[col] < col_mean - 4*col_std).sum()
                outliers_upper = (df[col] > col_mean + 4*col_std).sum()
                total_outliers = outliers_lower + outliers_upper
                
                if total_outliers > 0:
                    violations[col] = {
                        "outliers": int(total_outliers),
                        "percentage": float(total_outliers / len(df) * 100),
                        "range": [float(col_min), float(col_max)],
                        "mean": float(col_mean),
                        "std": float(col_std),
                        "lower_outliers": int(outliers_lower),
                        "upper_outliers": int(outliers_upper)
                    }
            
            if df[col].nunique() == 1:
                violations[col] = violations.get(col, {})
                violations[col]["constant_value"] = True
                violations[col]["value"] = float(col_min)
        
        findings = {
            "violations": violations, 
            "pass": len(violations) == 0,
            "columns_checked": len(numeric_cols)
        }
        
        context = self.retrieve_context("range validation numeric bounds outliers")
        prompt = self.create_validation_prompt("Range Check", context, findings)
        analysis = self.llm_client.query(prompt)
        
        return {"findings": findings, "ai_analysis": analysis}
    
    def validate_7_null_value_check(self, df: pd.DataFrame) -> Dict[str, Any]:
        """7. Null Value Check"""
        null_report = {}
        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                null_report[col] = {
                    "count": int(null_count),
                    "percentage": float(null_count / len(df) * 100)
                }
        
        findings = {
            "null_report": null_report,
            "total_nulls": sum(v['count'] for v in null_report.values()),
            "pass": len(null_report) == 0
        }
        
        context = self.retrieve_context("null value missing data detection")
        prompt = self.create_validation_prompt("Null Value Check", context, findings)
        analysis = self.llm_client.query(prompt)
        
        return {"findings": findings, "ai_analysis": analysis}
    
    def validate_12_duplicate_detection(self, df: pd.DataFrame) -> Dict[str, Any]:
        """12. Duplicate Detection"""
        total_duplicates = df.duplicated().sum()
        
        findings = {
            "total_duplicates": int(total_duplicates),
            "duplicate_percentage": float(total_duplicates / len(df) * 100),
            "pass": total_duplicates == 0
        }
        
        context = self.retrieve_context("duplicate detection record identification")
        prompt = self.create_validation_prompt("Duplicate Detection", context, findings)
        analysis = self.llm_client.query(prompt)
        
        return {"findings": findings, "ai_analysis": analysis}
    
    def validate_14_outlier_detection(self, df: pd.DataFrame) -> Dict[str, Any]:
        """14. Statistical Outlier Detection"""
        outliers = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:10]:
            try:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                outlier_mask_moderate = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
                outlier_mask_extreme = (df[col] < (Q1 - 3 * IQR)) | (df[col] > (Q3 + 3 * IQR))
                
                moderate_count = outlier_mask_moderate.sum()
                extreme_count = outlier_mask_extreme.sum()
                
                if moderate_count > 0:
                    outliers[col] = {
                        "moderate_outliers": int(moderate_count),
                        "extreme_outliers": int(extreme_count),
                        "moderate_percentage": float(moderate_count / len(df) * 100),
                        "extreme_percentage": float(extreme_count / len(df) * 100),
                        "Q1": float(Q1),
                        "Q3": float(Q3),
                        "IQR": float(IQR),
                        "severity": "HIGH" if extreme_count > len(df) * 0.01 else "MEDIUM"
                    }
            except:
                pass
        
        total_moderate = sum(v['moderate_outliers'] for v in outliers.values())
        total_extreme = sum(v['extreme_outliers'] for v in outliers.values())
        
        findings = {
            "outliers": outliers,
            "total_moderate_outliers": total_moderate,
            "total_extreme_outliers": total_extreme,
            "columns_with_outliers": len(outliers),
            "pass": total_extreme == 0
        }
        
        context = self.retrieve_context("outlier detection statistical anomalies IQR")
        prompt = self.create_validation_prompt("Outlier Detection", context, findings)
        analysis = self.llm_client.query(prompt)
        
        return {"findings": findings, "ai_analysis": analysis}
    
    def validate_15_data_integrity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """15. Data Integrity and Business Logic Validation"""
        integrity_issues = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_lower = col.lower()
            
            should_be_positive = any(keyword in col_lower for keyword in 
                                    ['price', 'cost', 'amount', 'quantity', 'volume', 
                                     'count', 'total', 'sum', 'age', 'distance', 'length'])
            
            if should_be_positive:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    integrity_issues[col] = {
                        "issue": "negative_values",
                        "count": int(negative_count),
                        "percentage": float(negative_count / len(df) * 100),
                        "severity": "CRITICAL",
                        "sample_values": df[df[col] < 0][col].head(5).tolist()
                    }
            
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                integrity_issues[f"{col}_infinity"] = {
                    "issue": "infinite_values",
                    "count": int(inf_count),
                    "percentage": float(inf_count / len(df) * 100),
                    "severity": "CRITICAL"
                }
            
            nan_count = df[col].isna().sum()
            if nan_count > len(df) * 0.1:
                integrity_issues[f"{col}_missing"] = {
                    "issue": "excessive_missing",
                    "count": int(nan_count),
                    "percentage": float(nan_count / len(df) * 100),
                    "severity": "HIGH"
                }
        
        string_cols = df.select_dtypes(include=['object']).columns
        
        for col in string_cols[:10]:
            empty_strings = (df[col].notna() & (df[col].astype(str).str.strip() == '')).sum()
            if empty_strings > 0:
                integrity_issues[f"{col}_empty_strings"] = {
                    "issue": "empty_strings",
                    "count": int(empty_strings),
                    "percentage": float(empty_strings / len(df) * 100),
                    "severity": "MEDIUM"
                }
            
            if df[col].notna().any():
                lengths = df[col].dropna().astype(str).str.len()
                if lengths.nunique() == 1 and len(df) > 10:
                    integrity_issues[f"{col}_uniform_length"] = {
                        "issue": "suspicious_uniform_length",
                        "length": int(lengths.iloc[0]),
                        "severity": "LOW"
                    }
        
        findings = {
            "integrity_issues": integrity_issues,
            "total_issues": len(integrity_issues),
            "critical_issues": sum(1 for v in integrity_issues.values() if v.get("severity") == "CRITICAL"),
            "pass": len(integrity_issues) == 0
        }
        
        context = self.retrieve_context("data integrity validation business logic")
        prompt = self.create_validation_prompt("Data Integrity Check", context, findings)
        analysis = self.llm_client.query(prompt)
        
        return {"findings": findings, "ai_analysis": analysis}
    
    def validate_24_quality_scoring(self, df: pd.DataFrame) -> Dict[str, Any]:
        """24. Production-Level Quality Scoring with Strict Standards"""
        scores = {}
        penalty_log = []
        
        overall_score = 100.0
        
        # COMPLETENESS SCORING
        total_cells = df.shape[0] * df.shape[1]
        null_cells = df.isnull().sum().sum()
        null_percentage = (null_cells / total_cells * 100) if total_cells > 0 else 0
        
        if null_percentage > 0:
            if null_percentage < 1:
                completeness_penalty = null_percentage * 5
            elif null_percentage < 5:
                completeness_penalty = 5 + (null_percentage - 1) * 8
            else:
                completeness_penalty = 37 + (null_percentage - 5) * 2
            
            overall_score -= min(completeness_penalty, 25)
            penalty_log.append(f"Completeness: -{completeness_penalty:.1f} points ({null_percentage:.2f}% missing data)")
        
        completeness = 100 - (null_percentage * 10)
        scores['completeness'] = max(0, float(completeness))
        
        # UNIQUENESS SCORING
        total_rows = len(df)
        duplicate_rows = df.duplicated().sum()
        duplicate_percentage = (duplicate_rows / total_rows * 100) if total_rows > 0 else 0
        
        if duplicate_rows > 0:
            if duplicate_percentage < 1:
                uniqueness_penalty = duplicate_percentage * 8
            elif duplicate_percentage < 5:
                uniqueness_penalty = 8 + (duplicate_percentage - 1) * 5
            else:
                uniqueness_penalty = 28 + (duplicate_percentage - 5) * 1.5
            
            overall_score -= min(uniqueness_penalty, 20)
            penalty_log.append(f"Uniqueness: -{uniqueness_penalty:.1f} points ({duplicate_rows} duplicates)")
        
        uniqueness = 100 - (duplicate_percentage * 15)
        scores['uniqueness'] = max(0, float(uniqueness))
        
        # VALIDITY SCORING
        validity_issues = 0
        total_validity_checks = 0
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            total_validity_checks += 1
            
            if 'price' in col.lower() or 'volume' in col.lower() or 'count' in col.lower():
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    validity_issues += 1
                    penalty = min(negative_count / len(df) * 20, 5)
                    overall_score -= penalty
                    penalty_log.append(f"Validity: -{penalty:.1f} points ({col} has {negative_count} negative values)")
            
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                validity_issues += 1
                penalty = min(inf_count / len(df) * 15, 4)
                overall_score -= penalty
                penalty_log.append(f"Validity: -{penalty:.1f} points ({col} has {inf_count} infinite values)")
        
        string_cols = df.select_dtypes(include=['object']).columns
        for col in string_cols:
            total_validity_checks += 1
            empty_count = (df[col].astype(str).str.strip() == '').sum()
            if empty_count > 0 and empty_count != df[col].isnull().sum():
                validity_issues += 1
                penalty = min(empty_count / len(df) * 10, 3)
                overall_score -= penalty
                penalty_log.append(f"Validity: -{penalty:.1f} points ({col} has {empty_count} empty strings)")
        
        validity = 100 - (validity_issues * 10) if total_validity_checks > 0 else 100
        scores['validity'] = max(0, float(validity))
        
        # CONSISTENCY SCORING
        consistency_issues = 0
        
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    numeric_conversion = pd.to_numeric(df[col], errors='coerce')
                    conversion_rate = numeric_conversion.notna().sum() / len(df)
                    if 0.7 < conversion_rate < 0.95:
                        consistency_issues += 1
                        penalty = 3
                        overall_score -= penalty
                        penalty_log.append(f"Consistency: -{penalty:.1f} points ({col} has mixed types)")
                except:
                    pass
        
        consistency = 100 - (consistency_issues * 15)
        scores['consistency'] = max(0, float(consistency))
        
        # OUTLIER PENALTY
        outlier_penalty = 0
        for col in numeric_cols[:10]:
            try:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_mask = (df[col] < (Q1 - 3 * IQR)) | (df[col] > (Q3 + 3 * IQR))
                outlier_count = outlier_mask.sum()
                
                if outlier_count > len(df) * 0.05:
                    penalty = min((outlier_count / len(df)) * 8, 2)
                    outlier_penalty += penalty
                    penalty_log.append(f"Outliers: -{penalty:.1f} points ({col} has {outlier_count} extreme outliers)")
            except:
                pass
        
        overall_score -= min(outlier_penalty, 10)
        
        # PATTERN VIOLATIONS
        for col in string_cols[:5]:
            unique_ratio = df[col].nunique() / len(df)
            
            if unique_ratio > 0.95 and len(df) > 100:
                penalty = 2
                overall_score -= penalty
                penalty_log.append(f"Pattern: -{penalty:.1f} points ({col} has suspiciously high uniqueness)")
            
            if unique_ratio < 0.01 and len(df) > 100:
                penalty = 2
                overall_score -= penalty
                penalty_log.append(f"Pattern: -{penalty:.1f} points ({col} has very low variance)")
        
        overall_score = max(0, min(100, overall_score))
        scores['overall_quality_score'] = float(overall_score)
        scores['penalty_details'] = penalty_log
        scores['issues_found'] = len(penalty_log)
        
        findings = {
            "quality_scores": scores,
            "null_percentage": float(null_percentage),
            "duplicate_percentage": float(duplicate_percentage),
            "validity_issues": validity_issues,
            "consistency_issues": consistency_issues
        }
        
        context = self.retrieve_context("quality scoring data health metrics")
        prompt = self.create_validation_prompt("Quality Scoring", context, findings)
        analysis = self.llm_client.query(prompt)
        
        self.quality_score = overall_score
        
        return {"findings": findings, "ai_analysis": analysis}
    
    def run_selected_validations(self, df: pd.DataFrame, selected_validations: List[str], 
                                 progress_callback=None, status_callback=None) -> Dict[str, Any]:
        """Run selected validation techniques"""
        validation_map = {
            "Data Type Check": self.validate_1_data_type_check,
            "Range Check": self.validate_2_range_check,
            "Null Value Check": self.validate_7_null_value_check,
            "Duplicate Detection": self.validate_12_duplicate_detection,
            "Outlier Detection": self.validate_14_outlier_detection,
            "Data Integrity Check": self.validate_15_data_integrity,
            "Quality Scoring": self.validate_24_quality_scoring,
        }
        
        results = {}
        total = len(selected_validations)
        
        for idx, validation_name in enumerate(selected_validations):
            if status_callback:
                status_callback(f"Running: {validation_name}...")
            
            if validation_name in validation_map:
                try:
                    result = validation_map[validation_name](df)
                    results[validation_name] = result
                except Exception as e:
                    results[validation_name] = {"error": str(e)}
            
            if progress_callback:
                progress_callback((idx + 1) / total)
        
        if status_callback:
            status_callback("✅ Validation Complete!")
        
        self.validation_results = results
        return results