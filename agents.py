#agents.py
from llama_index.llms.openai import OpenAI
from llama_index.core.indices.query.query_transform.base import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.postprocessor import LLMRerank
from llama_index.core import QueryBundle
import json
from config import ModelConfig
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from utils import JSONExtractor

@dataclass
class PatientInfo:
    """Structured patient information extracted from transcript"""
    name: str
    age: float  
    weight: float  
    symptoms: List[str]
    severity: str  
    allergies: List[str]
    current_medications: List[str]  
    pmhx: List[str]  
    raw_text: str 

@dataclass
class MedicationDose:
    """Structured medication  information"""
    medication: str
    dose: float
    unit: str
    route: str
    frequency: str
    max_dose: Optional[float] = None
    calculation_method: str = ""
    source: str = ""

@dataclass
class ClinicalRecommendation:
    """Simplified clinical recommendation"""
    patient_info: PatientInfo
    management_plan: str
    cks_guideline_content: str
    local_guideline_content: str
    cks_medication_dose: Optional[MedicationDose]
    local_medication_dose: Optional[MedicationDose]
    cks_manual_dose: str  
    local_manual_dose: str  
    safety_alerts: List[Dict[str, str]]

@dataclass
class ClinicalQuery:
    """Structured clinical query """
    question: str
    question_type: str  
    focus_area: str  
    raw_query: str

class TranscriptStandardiser:
    """Agent to extract structured pt info from transcript"""
    
    def __init__(self):
        self.llm = ModelConfig.get_llm("llama-4-17b")

    def parse_transcript(self, transcript):
        prompt = f"""
        You are a medical data extraction assistant. Extract patient information from this clinical transcript and return it as valid JSON.

        Clinical Transcript:
        {transcript}

        Extract these fields and return ONLY valid JSON with actual values (not variable names):

        - name: Extract the patient's name (if not found, use "Unknown Patient")
        - age: Extract age in years as a number (if not found, use 0)
        - weight: Extract weight in kg as a number (if not found, use 0)  
        - symptoms: Extract all symptoms as an array of strings (if none found, use [])
        - severity: Determine severity as "mild", "moderate", or "severe" based on clinical presentation (if unclear, use "moderate")
        - allergies: Extract any allergies as an array of strings (if absence of allergies confirmed return ["No known drug allergies"], else if not detailed return [])
        - current_medications: Extract any current medications as an array of strings (if none found, use [])
        - pmhx: Extract any previous medical history as an array of strings (if none found, use [])

        Example response format:
        {{"name": "John Smith", "age": 5, "weight": 18.5, "symptoms": ["cough", "fever"], "severity": "moderate", "allergies": [], "current_medications": ["paracetamol 500mg", "ibuprofen 400mg"], "pmhx": ["asthma", "eczema"]}}

        Return only the JSON object with actual extracted values from the transcript, without further comments:
        """
        try:
            response = self.llm.complete(prompt)
            json_str = JSONExtractor.extract_json(response.text)
            if not json_str:
                print("Failed to extract valid JSON from LLM ")
                print(f" response: {response.text}")
                return None

            data = json.loads(json_str) 
            print(f"DEBUG: LLM response: {data}")
            return PatientInfo(
                name=data.get('name', 'Unknown'),
                age=float(data.get('age', 0)),
                weight=float(data.get('weight', 0)),
                symptoms=data.get('symptoms', []),
                severity=data.get('severity', 'moderate'),
                allergies=data.get('allergies', []),
                current_medications=data.get('current_medications', []), 
                pmhx=data.get('pmhx', []),  
                raw_text=transcript
            )

        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Attempted to parse: {json_str if 'json_str' in locals() else 'No JSON extracted'}")
            return None
        except Exception as e:
            print(f"Error parsing transcript: {e}")
            return None

class ClinicalSummariser:
    """Agent to create concise clinical summaries from transcripts"""
    
    def __init__(self):
        self.llm = ModelConfig.get_llm("llama-4-17b")
    
    def create_clinical_summary(self, patient_info, raw_transcript):
        """Create a structured clinical summary from patient info and transcript"""
        
        prompt = f"""You are a clinical summarisation specialist. Create a concise, structured clinical summary from this information.

        **Original Transcript:**
        {raw_transcript}

        Create a clinical summary that:
        1. Preserves all clinically relevant details
        2. Is concise but comprehensive
        3. Uses proper medical terminology
        4. Maintains chronological order of events
        5. Includes key examination findings
        6. Preserves assessment and initial plan

        Format as a structured clinical summary (4-6 sentences maximum):"""
        
        try:
            response = self.llm.complete(prompt)
            summary = response.text.strip()
            
            print(f"Clinical summary created ({len(summary)} chars vs {len(raw_transcript)} original)")
            return summary
            
        except Exception as e:
            print(f"Clinical summarisation failed: {e}")
            return raw_transcript

class QueryStandardiser:
    """Transform clinical query using hypothetical document embedding to improve retrieval"""
    def __init__(self, use_hyde=True):
        self.llm = ModelConfig.get_llm("llama-4-17b")
        self.use_hyde = use_hyde
        self.hyde_transform = None
        self._hyde_initialised = False
    
    def _init_hyde(self):
        if self.use_hyde and not self._hyde_initialised:
            try:
                from llama_index.core import Settings
                #ensure global setting set to qwen otherwise error to oepn ai
                qwen_embed = ModelConfig.get_embedding_model("qwen")
                llama_llm = ModelConfig.get_llm("llama-4-17b")
                
                Settings.embed_model = qwen_embed
                Settings.llm = llama_llm
                
                print(f"Creating HyDE with LLM {type(llama_llm).__name__} and embed model {type(qwen_embed).__name__}")
                
                # Create huyde with explicit specification of qwen 
                self.hyde_transform = HyDEQueryTransform(
                    llm=llama_llm,  # 
                    include_original=True
                )
                self._hyde_initialised = True
                print("HyDE transform created successfully with custom models")
                
            except Exception as e:
                print(f"Failed to create HyDE transform: {e}")
                self.hyde_transform = None
                self.use_hyde = False
    def standardise_query(self, patient_info, clinical_context, clinical_question=None):
        """standardisation of clinical question, then HyDE """
        if self.use_hyde:
            if self.use_hyde and not self._hyde_initialised:
                self._init_hyde()
        
        # Create  query fo hyde
        primary_query = self._create_query(patient_info, clinical_context, clinical_question)
        
        return {
            'primary_query': primary_query,
            'hyde_transform': self.hyde_transform,
            'patient_info': patient_info,
            'age_group': 'paediatric' if patient_info.age < 18 else 'adult',
            'clinical_question': clinical_question,
            'raw_context': clinical_context
        }
    
    def _create_query(self, patient_info, clinical_context, clinical_question=None):
        """Create query with  standardisation of clinical question"""
        
        if clinical_question:
            standardised_question = self._standardise_question(clinical_question, clinical_context)
            
            # Question + Patient context
            primary_query = f"{standardised_question}\n\nPatient context: {patient_info.age}y {patient_info.weight}kg with {', '.join(patient_info.symptoms[:3])}"
            
            # Question + Full context
            # primary_query = f"{standardised_question}\n\n{clinical_context}"
            
            print(f"Primary query: {standardised_question}")
            return primary_query
        else:
            # No specific question - use clinical context 
            return clinical_context
    
    def _standardise_question(self, clinical_question, clinical_context):
            #Summary is slightly verbose
            """ standardisation: fix typos, expand abbreviations, clarify"""
            
            prompt = f"""Standardise this clinical question by fixing typos and making it clear in the context of the patient's background. Keep it concise.

            Original question: {clinical_question}

            Patient background: {clinical_context}

            Return the standardised question (one sentence, use medical terminology, clearly relate to background, diagnosis, differentials, management, medication):"""
            
            try:
                response = self.llm.complete(prompt)
                standardised = response.text.strip().strip('"').strip("'")
                
                # Fallback to original if fails
                if not standardised or len(standardised) < 3:
                    return clinical_question
                    
                return standardised
                
            except Exception as e:
                print(f"Question standardisation failed: {e}")
                return clinical_question
    
class RetrievalAgent:
    """Agent to retrieve relevant information from the knowledge layer"""
    def __init__(self, cks_index, local_index, use_llm_reranking = True, use_hyde = True):
        self.cks_index = cks_index
        self.local_index = local_index
        self.llm = ModelConfig.get_llm("llama-4-17b") 
        self.use_llm_reranking = use_llm_reranking
        self.use_hyde = use_hyde
        
        #  LLMRerank with qwen 
        try:
            self.llm_reranker = LLMRerank(
                llm=self.llm,  
                choice_batch_size=5,  
                top_n=5  
            ) if use_llm_reranking else None
            if use_llm_reranking:
                print("LLMReranking")
        except ImportError as e:
            print(f"LLMRerank not available: {e}")
            self.llm_reranker = None
            self.use_llm_reranking = False

    def retrieve_guideline(self, query_info, top_k=5, retrieval_k=None):
        """Retrieve guidelines with optional HyDE and LLM reranking"""
        primary_query = query_info['primary_query']
        hyde_transform = query_info['hyde_transform'] if self.use_hyde else None
        
        print(f"Retrieving {retrieval_k} candidates - {top_k} final results")
        print(f"Features: HyDE {self.use_hyde} , LLM Reranking {self.use_llm_reranking}")
        if top_k is None:
            top_k = 5
        # Calculate retrieval size
        if retrieval_k is None:
            retrieval_k = top_k * 2 if self.use_llm_reranking else top_k

        
        # Retrieve from CKS index
        if hyde_transform:
            print("Applying HyDE transformation...")
            cks_query_engine = TransformQueryEngine(
                self.cks_index.as_query_engine(similarity_top_k=retrieval_k), 
                hyde_transform
            )
            cks_response = cks_query_engine.query(primary_query)
            cks_nodes = cks_response.source_nodes
        else:
            cks_retriever = self.cks_index.as_retriever(similarity_top_k=retrieval_k)
            cks_nodes = cks_retriever.retrieve(primary_query)
        
        # Retrieve from local index
        local_nodes = []
        if self.local_index:
            if hyde_transform:
                local_query_engine = TransformQueryEngine(
                    self.local_index.as_query_engine(similarity_top_k=retrieval_k), 
                    hyde_transform
                )
                local_response = local_query_engine.query(primary_query)
                local_nodes = local_response.source_nodes       
            else:
                local_retriever = self.local_index.as_retriever(similarity_top_k=retrieval_k)
                local_nodes = local_retriever.retrieve(primary_query)
        
        # Apply LLM reranking if True
        if self.use_llm_reranking and self.llm_reranker:
            print(f"LLM reranking {len(cks_nodes)} CKS w/ {len(local_nodes)} local - {top_k} each")
            cks_nodes = self._llm_rerank_nodes(cks_nodes, primary_query, top_k)
            local_nodes = self._llm_rerank_nodes(local_nodes, primary_query, top_k)
        else:
            print("Using embedding-only retrieval")
            cks_nodes = cks_nodes[:top_k]
            local_nodes = local_nodes[:top_k]

        return {
            'cks': cks_nodes,
            'local': local_nodes
        }
    
    def _llm_rerank_nodes(self, nodes, query, top_k):
        """Apply LLM-based reranking using LlamaIndex's LLMRerank"""
        if not nodes or not self.llm_reranker:
            return nodes[:top_k]
            
        try:
            query_bundle = QueryBundle(query_str=query)
            
            self.llm_reranker.top_n = min(top_k, len(nodes))
            
            # Apply LLM reranking
            reranked_nodes = self.llm_reranker.postprocess_nodes(nodes, query_bundle)
            
            print(f"LLM reranked {len(nodes)} to {len(reranked_nodes)} nodes")
            return reranked_nodes
            
        except Exception as e:
            print(f"LLM reranking failed: {e}, using embedding order")
            return nodes[:top_k]
    
    
class ConflictDetector:
    """Detect descrepancies between cks and local guidelines"""
    def __init__(self):
        self.llm = ModelConfig.get_llm("llama-4-17b")

    def detect_conflicts(self, cks_content, local_content, critical_fields):
        prompt = f"""
        Compare these two clinical guidelines and identify any conflicts or differences. 
        Focus especially on the following critical fields: {', '.join(critical_fields)}
        
        NICE National Guideline:
        {cks_content[:2000]}
        
        Local Hospital Guideline:
        {local_content[:2000]}
        
        Return conflicts in the following JSON object:
        {{
            "section": "what aspect of guideline differs (diagnosis criteria/management/medication doses etc)",
            "nice_recommendation": "what NICE CKS says",
            "local_recommendation": "what local says",
            "severity": "minor/moderate/critical - potential for patient harm",
            "clinical_impact": "brief description of impact"
        }}
        
        Return empty array [] if no conflicts. Strictly return the JSON, with no other text or comments.
        """
    
        try:
            response = self.llm.complete(prompt)
            json_str = JSONExtractor.extract_json(response.text)
            if not json_str:
                print("No JSON found in conflict detection response")
                return []
            
            conflicts = json.loads(json_str)
            return conflicts if isinstance(conflicts, list) else []
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed in conflict detection: {e}")
            return []
        except Exception as e:
            print(f"Conflict detection failed: {e}")
            return []
        
class DoseCalculator:
    """agent to calculate medication doses"""
    def __init__(self):
        self.llm = ModelConfig.get_llm("llama-4-17b")

    def parse_dosing_info(self, guideline_content, weight):
        prompt = f"""
        Extract dosing information from this guideline text and calculate the dose for a patient weighing {weight} kg.
        
        Guideline text:
        {guideline_content}
        
        Return JSON with:
        {{
            "medication": "medication name",
            "dose_per_kg": number or null,
            "calculated_dose": number (calcualte the dose {weight}kg),
            "unit": "mg (convert if required)",
            "max_dose": number or null,
            "min_dose": number or null,
            "route": "oral/iv/im/nebulised/etc",
            "frequency": "how often to give",
            "calculation_details": "show how you calculated it",
            "interactions": ["array of interactions with other drugs, [] if none specified"]
        }}
        
        If no dosing information found, return null.
        If dose is given as absolute (not per kg), just return that dose.
        Return only a valid JSON:
        """
        try:
            response = self.llm.complete(prompt)
            json_str = JSONExtractor.extract_json(response.text)
            if not json_str:
                print("No JSON found in dosing response")
                return None, 'N/A'
            
            data = json.loads(json_str) 
            llm_calculated_dose = data.get('calculated_dose', 0)
            if data.get('max_dose') and llm_calculated_dose > data.get('max_dose'):
                llm_calculated_dose = data.get('max_dose')
                data['calculation_details'] += f" (capped at max dose)"
            if data.get('min_dose') and llm_calculated_dose < data['min_dose']:
                llm_calculated_dose = data['min_dose']
                data['calculation_details'] += f" (increased to min dose)"
            
            # Check for calculation discrepancies
            manual_dose = 'N/A'
            if data.get('dose_per_kg'):
                manual_calculation = data.get('dose_per_kg') * weight
                if abs(manual_calculation - llm_calculated_dose) > 0.1:
                    manual_dose = str(manual_calculation)
                    data['calculation_details'] += f" (MISMATCH - LLM calculated dose {llm_calculated_dose}mg does not match manual calculation {manual_calculation}mg)"

            medication_dose = MedicationDose(
                medication=data.get('medication', 'unknown'),
                dose=llm_calculated_dose,
                unit=data.get('unit', 'mg'),
                route=data.get('route', 'not specified'),
                frequency=data.get('frequency', 'as directed'),
                max_dose=data.get('max_dose'),
                calculation_method=data.get('calculation_details', 'extracted from guideline'),
                source='guideline_extraction'
            )
            return medication_dose, manual_dose

        except json.JSONDecodeError as e:
            print(f"JSON parsing failed in dose calculation: {e}")
            return None, 'N/A'
        except Exception as e:
            print(f"Error parsing dosing info: {e}")
            return None, 'N/A'

        
class SynthesisAgent:
    """Syntehsise final clinical reccomendation"""
    def __init__(self):
        self.llm = ModelConfig.get_llm("llama-4-17b")
        self.dose_calculator = DoseCalculator()

    def _generate_management_plan(self, patient_info, cks_text, local_text, conflicts):
        """Generates a management plan using all available context."""
        prompt = f"""
        You are a clinical decision support assistant. Your task is to synthesise a comprehensive management plan for a patient based on the provided information below. Regional and local knowledge bases have been queried (NICE CKS and local guidelines), so prioritise this information in the generation of your plan. Prioritise clinical safety and evidence-based recommendations.

        Clinical History:
        ---
        {patient_info.raw_text}
        ---

        **Structured Patient Summary:**
        - Name: {patient_info.name}
        - Age: {patient_info.age} years
        - Weight: {patient_info.weight} kg
        - Key Symptoms: {', '.join(patient_info.symptoms)}
        - Assessed Severity: {patient_info.severity}
        - Allergies: {', '.join(patient_info.allergies) if patient_info.allergies else 'None documented'}
        - Current Medications: {', '.join(patient_info.current_medications) if patient_info.current_medications else 'None documented'}
        - Previous Medical History: {', '.join(patient_info.pmhx) if patient_info.pmhx else 'None documented'}

        **Retrieved NICE CKS Guidelines:**
        {cks_text}

        **Retrieved Local Guidelines:**
        {local_text}

        **Guideline Conflicts Detected:**
        {json.dumps(conflicts, indent=2) if conflicts else "No significant conflicts detected."}

        Provide a comprehensive, evidence-based management plan that:
        1. Acknowledges the clinical presentation from the transcript
        2. References specific guideline recommendations
        3. Considers the patient's current medications and medical history for drug interactions and contraindications
        4. Addresses any conflicts between guidelines with clinical reasoning, and need for manual verification
        5. Includes specific medication dosing, monitoring, and safety considerations
        6. Provides clear next steps and follow-up recommendations
        
        Format as a structured clinical management plan.
        """
        try:
            response = self.llm.complete(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating management plan: {e}")
            return "Failed to generate a management summary due to an error."
    
    def synthesise_recs(self, patient_info, guidelines, conflicts):
        """Synthesise info into recommendation for usr"""
        cks_content = guidelines.get('cks', [])
        local_content = guidelines.get('local', [])
        
            # Combine all retrieved content
        separator = "\n\n-Guideline Excerpt-\n\n"
        cks_text = separator.join([node.text for node in cks_content]) if cks_content else "No NICE CKS guidelines found"
        local_text = separator.join([node.text for node in local_content]) if local_content else "No local guidelines found"
        
        # Generate  management plan
        management_plan = self._generate_management_plan(patient_info, cks_text, local_text, conflicts)
        
        # Calculate doses using combined info
        cks_dose, cks_manual = None, 'N/A'
        local_dose, local_manual = None, 'N/A'
        
        if cks_content:
            combined_cks_text = "\n\n".join([node.text for node in cks_content])
            cks_dose, cks_manual = self.dose_calculator.parse_dosing_info(combined_cks_text, patient_info.weight)
        
        if local_content:
            combined_local_text = "\n\n".join([node.text for node in local_content])
            local_dose, local_manual = self.dose_calculator.parse_dosing_info(combined_local_text, patient_info.weight)
        
        safety_alerts = self._generate_safety_alerts(cks_dose, local_dose, cks_manual, local_manual, conflicts, patient_info)
        
        return ClinicalRecommendation(
            patient_info=patient_info,
            cks_guideline_content=cks_text,  
            local_guideline_content=local_text,  
            cks_medication_dose=cks_dose,
            local_medication_dose=local_dose,
            cks_manual_dose=cks_manual,
            local_manual_dose=local_manual,
            safety_alerts=safety_alerts,
            management_plan=management_plan  
        )

    
    def _generate_safety_alerts(self, cks_dose, local_dose, cks_manual, local_manual, conflicts, patient_info):
        """Generate safety alerts for discrepancies and missing information"""
        alerts = []
        
        # Add disclaimer always
        alerts.append({
            'level': 'blue',
            'message': 'This tool is for R&D only and should not be used for clinical care'
        })
        
        # Check for dose calculation discrepancies
        if cks_manual != 'N/A':
            alerts.append({
                'level': 'red',
                'message': f'NICE dose calculation discrepancy detected - manual verification required'
            })
        
        if local_manual != 'N/A':
            alerts.append({
                'level': 'red',
                'message': f'Local dose calculation discrepancy detected - manual verification required'
            })
        
        #Check for conflicts between cks and local doses
        if cks_dose and local_dose:
            if cks_dose.medication.lower() == local_dose.medication.lower():
                if abs(cks_dose.dose - local_dose.dose) > 0.1:
                    alerts.append({
                        'level': 'red',
                        'message': f'Dose conflict: NICE recommends {cks_dose.dose}{cks_dose.unit} vs Local {local_dose.dose}{local_dose.unit}'
                    })
                if cks_dose.route != local_dose.route:
                    alerts.append({
                        'level': 'amber',
                        'message': f'Route conflict: NICE recommends {cks_dose.route} vs Local {local_dose.route}'
                    })
        
        # Check for guideline conflicts
        critical_conflicts = [c for c in conflicts if c.get('severity') == 'critical']
        if critical_conflicts:
            alerts.append({
                'level': 'red',
                'message': f'Critical guideline conflict: {critical_conflicts[0].get("section", "unknown area")}'
            })
        
        moderate_conflicts = [c for c in conflicts if c.get('severity') == 'moderate']
        if moderate_conflicts:
            alerts.append({
                'level': 'amber',
                'message': f'Guideline variation noted: {moderate_conflicts[0].get("section", "unknown area")}'
            })
        
        # prompt interactions
        if patient_info.current_medications and (cks_dose or local_dose):
            alerts.append({
                'level': 'amber',
                'message': f'Patient on current medications: {", ".join(patient_info.current_medications)} - check for interactions'
            })
        
        # Check allergy status 
        if not patient_info.allergies:
            alerts.append({
                'level': 'amber',
                'message': 'Allergy status not documented - verify before medication administration'
            })
        elif patient_info.allergies == ["No known drug allergies"]:
            alerts.append({
                'level': 'blue',
                'message': 'No known drug allergies documented'
            })
        else:
            alerts.append({
                'level': 'amber',
                'message': f'Known allergies: {", ".join(patient_info.allergies)} - check compatibility with prescribed medications'
            })
        
        #Medical history alerts
        if patient_info.pmhx:
            alerts.append({
                'level': 'blue',
                'message': f'Previous medical history: {", ".join(patient_info.pmhx)} - consider in treatment decisions'
            })
        
        # Check for missing patient information
        if not patient_info.weight or patient_info.weight == 0:
            alerts.append({
                'level': 'red',
                'message': 'Patient weight missing - required for dose calculations'
            })
        
        # Check for missing dosing information
        if not cks_dose and not local_dose:
            alerts.append({
                'level': 'amber',
                'message': 'No medication dosing information found in guidelines'
            })
        
        return alerts