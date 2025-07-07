import os
from typing import Optional
from agents import (
    TranscriptStandardiser, 
    ClinicalSummariser,
    QueryStandardiser, 
    RetrievalAgent, 
    ConflictDetector, 
    SynthesisAgent,
    ClinicalRecommendation,
    PatientInfo
)
from knowledge_layer import KnowledgeLayer

class ClinicalDecisionSupport:
    """
    Class to orchestrate the CDS pipeline
    """
    def __init__(self, cks_path = "cks_conditions_data.json", local_dir= "/paed-guidelines", cache_dir= "./guideline_cache", use_cache = True, use_llm_reranking = True, use_hyde = True, use_summary=True):
        
        from config import ModelConfig
        ModelConfig.initialise_global_settings()
        self.knowledge_layer = KnowledgeLayer(use_cache=use_cache, cache_dir=cache_dir)
        #Initialise agents
        self.transcript_parser = TranscriptStandardiser()
        self.clinical_summariser = ClinicalSummariser()
        self.query_standardizer = QueryStandardiser(use_hyde=use_hyde)
        self.conflict_detector = ConflictDetector()
        self.synthesis_agent = SynthesisAgent()
        
        # Store config
        self.use_llm_reranking = use_llm_reranking
        self.use_hyde = use_hyde
        self.use_summary = use_summary
        self.default_top_k = 3 
        if use_llm_reranking:
            self.default_retrieval_k = 6 
        else:
            self.default_retrieval_k = 3
        self.is_ready = False
        self._load_knowledge(cks_path, local_dir)
    
    def _load_knowledge(self, cks_path, local_dir):
        """Load and index guidelines"""
        try:
            print("Loading guidelines")
            cks_index, local_index = self.knowledge_layer.data_loader(cks_path, local_dir)
            self.retrieval_agent = RetrievalAgent(
                cks_index, 
                local_index, 
                use_llm_reranking=self.use_llm_reranking,
                use_hyde=self.use_hyde
            )
            self.is_ready = True
            features = []
            if self.use_hyde:
                features.append("HyDE")
            if self.use_llm_reranking:
                features.append("LLM Reranking")
            
            feature_str = f" with {', '.join(features)}" if features else " (basic embedding retrieval)"
            print(f"CDS system ready{feature_str}")
        except Exception as e:
            print(f"Error loading knowledge: {e}")
            self.is_ready = False
            raise
    
    def set_retrieval_params(self, top_k: Optional[int] = None, retrieval_k: Optional[int] = None):
        """Set default retrieval parameters for UI """
        if top_k is not None:
            self.default_top_k = top_k
            print(f"Using top_k: {top_k}")
            
        if retrieval_k is not None:
            self.default_retrieval_k = retrieval_k
            print(f"Using retrieval_k: {retrieval_k}")

    def get_retrieval_params(self):
        """Get current retrieval parameters for UI """
        return {
            'default_top_k': self.default_top_k,
            'default_retrieval_k': self.default_retrieval_k,
            'hyde_enabled': self.use_hyde,
            'llm_reranking_enabled': self.use_llm_reranking
        }

    def toggle_features(self, use_hyde: Optional[bool] = None, use_llm_reranking: Optional[bool] = None):
        """Toggle features without reloading knowledge"""
        if use_hyde is not None:
            self.use_hyde = use_hyde
            # Update query standardizer
            self.query_standardizer = QueryStandardiser(use_hyde=use_hyde)
            
        if use_llm_reranking is not None:
            self.use_llm_reranking = use_llm_reranking
            
        # Update retrieval agent if ready
        if self.is_ready and self.retrieval_agent:
            self.retrieval_agent.use_hyde = self.use_hyde
            self.retrieval_agent.use_llm_reranking = self.use_llm_reranking
            
        print(f"Features updated: HyDE {self.use_hyde}, LLM Reranking {self.use_llm_reranking}")

    def process_case(self, transcript, top_k = None, retrieval_k = None, clinical_question=None):
        """Process transcript and return reocmmendation"""
        if not self.is_ready:
            print('CDS system not ready - please check if knowledge is loaded')
            return None
        
        final_top_k = top_k or self.default_top_k
        final_retrieval_k = retrieval_k or self.default_retrieval_k
        try:
            # Parse transcript to retrieve structured patient info
            print('Stage 1- Parsing transcript')
            patient_info = self.transcript_parser.parse_transcript(transcript)
            if not patient_info:
                print("Failed to parse transcript - could not extract patient information")
            print(f"Extracted patient info: {patient_info.name}, Age: {patient_info.age}, Weight: {patient_info.weight}kg")

            # Standardise query
            print('Stage 2 - standardising query')
            if clinical_question:
                print(f"Clinical question: {clinical_question}")
            #clinical_context = ' '.join(patient_info.symptoms) if patient_info.symptoms else "clinical assessment"
            if self.use_summary:
                print('Creating clinical summary as context')
                clinical_context = self.clinical_summariser.create_clinical_summary(patient_info, transcript)
            else:
                print('Using full transcript as context')
                clinical_context = patient_info.raw_text
            
            query_info = self.query_standardizer.standardise_query(
                patient_info, 
                clinical_context,
                clinical_question 
            )
            #print(f'query context: {clinical_context}')
            print('using full transcript as context')

            #Retrieve guidelines
            print('Stage 3 - retrieving guidelines')
            guidelines = self.retrieval_agent.retrieve_guideline(query_info, top_k=top_k, retrieval_k=retrieval_k)
            cks_found = len(guidelines.get('cks', []))
            local_found = len(guidelines.get('local', []))
            print(f"Retrieved {cks_found} NICE CKS guidelines, {local_found} local guidelines")

            # Detect conflicts
            print('Stage 4 - detecting conflicts')
            conflicts = []
            separator = "\n\n-Guideline Excerpt-\n\n"
            cks_text = separator.join([node.text for node in guidelines.get('cks', [])])
            local_text = separator.join([node.text for node in guidelines.get('local', [])])

            if cks_text and local_text:
                conflicts = self.conflict_detector.detect_conflicts(
                    cks_text, local_text, ['dosing', 'management', 'criteria']
                )
                print(f"Found {len(conflicts)} potential conflicts")
            else:
                print("Insufficient guidelines for conflict detection")

            # Synthesise recommendation
            print('Stage 5 - synthesising recommendation')
            recommendation = self.synthesis_agent.synthesise_recs(patient_info, guidelines, conflicts)
            
            print("Clinical decision support system complete, producing recommendation")
            return recommendation
        except Exception as e:
            print(f"Error in clinical processing pipeline: {e}")
            raise
    
    def system_status(self) -> dict:
        """Get current system current status"""
        return {
            'ready': self.is_ready,
            'knowledge_loaded': self.retrieval_agent is not None,
            'cks_available': self.retrieval_agent.cks_index is not None if self.retrieval_agent else False,
            'local_available': self.retrieval_agent.local_index is not None if self.retrieval_agent else False,
            'features': {
                'hyde_enabled': self.use_hyde,
                'llm_reranking_enabled': self.use_llm_reranking,
                'llm_reranker_available': hasattr(self.retrieval_agent, 'llm_reranker') and self.retrieval_agent.llm_reranker is not None if self.retrieval_agent else False
            },
            'retrieval_params': {
                'default_top_k': self.default_top_k,
                'default_retrieval_k': self.default_retrieval_k
            }
        }
    
    def format_recommendation(self, recommendation):
        """Recommendation format, to change later to choose what to filter what to display"""
        
        critical_alerts = [alert for alert in recommendation.safety_alerts if alert['level'] == 'red']
        warning_alerts = [alert for alert in recommendation.safety_alerts if alert['level'] == 'amber']
        info_alerts = [alert for alert in recommendation.safety_alerts if alert['level'] == 'blue']
        
        cks_medication = None
        local_medication = None

        if recommendation.cks_medication_dose:
            cks_medication = {
                'medication': recommendation.cks_medication_dose.medication,
                'dose': f"{recommendation.cks_medication_dose.dose} {recommendation.cks_medication_dose.unit}",
                'route': recommendation.cks_medication_dose.route,
                'frequency': recommendation.cks_medication_dose.frequency,
                'calculation': recommendation.cks_medication_dose.calculation_method,
                'manual_check': recommendation.cks_manual_dose
            }
        
        if recommendation.local_medication_dose:
            local_medication = {
                'medication': recommendation.local_medication_dose.medication,
                'dose': f"{recommendation.local_medication_dose.dose} {recommendation.local_medication_dose.unit}",
                'route': recommendation.local_medication_dose.route,
                'frequency': recommendation.local_medication_dose.frequency,
                'calculation': recommendation.local_medication_dose.calculation_method,
                'manual_check': recommendation.local_manual_dose
            }
        
        return {
            'patient': {
                'name': recommendation.patient_info.name,
                'age': recommendation.patient_info.age,
                'weight': recommendation.patient_info.weight,
                'symptoms': recommendation.patient_info.symptoms,
                'severity': recommendation.patient_info.severity,
                'allergies': recommendation.patient_info.allergies,
                'current_medications': recommendation.patient_info.current_medications,  # NEW FIELD
                'pmhx': recommendation.patient_info.pmhx  # NEW FIELD
            },
            
            'medications': {
                'nice_cks': cks_medication,
                'local': local_medication
            },
            
            'safety_alerts': {
                'critical': critical_alerts,
                'warnings': warning_alerts,
                'info': info_alerts
            },
            
            'management_plan': recommendation.management_plan,
            'executive_summary': self._generate_summary(recommendation),
            
            'source_guidelines': {
                'nice_cks': recommendation.cks_guideline_content,
                'local': recommendation.local_guideline_content
            },
            
            'dev': {
                'retrieval_features': {
                    'hyde_used': self.use_hyde,
                    'llm_reranking_used': self.use_llm_reranking,
                    'llm_reranker_available': hasattr(self.retrieval_agent, 'llm_reranker') and self.retrieval_agent.llm_reranker is not None if self.retrieval_agent else False
                },
                'guidelines_retrieved': {
                    'cks_count': len(recommendation.cks_guideline_content.split('-Guideline Excerpt-')) if recommendation.cks_guideline_content else 0,
                    'local_count': len(recommendation.local_guideline_content.split('-Guideline Excerpt-')) if recommendation.local_guideline_content else 0
                }
            }
        }
    

    
    def _generate_summary(self, recommendation):
        """Generate a concise summary """
        patient = recommendation.patient_info
        
        summary_parts = []
        
        # Patient summary with additional context
        patient_context = f"{patient.name} ({patient.age}y, {patient.weight}kg)"
        if patient.pmhx:
            patient_context += f", PMHx: {', '.join(patient.pmhx[:2])}"  
        if patient.current_medications:
            patient_context += f", on {', '.join(patient.current_medications[:2])}"  
        
        summary_parts.append(f"{patient_context} presents with {patient.severity} {', '.join(patient.symptoms)}")
        
        if recommendation.cks_medication_dose:
            med = recommendation.cks_medication_dose
            summary_parts.append(f"Recommended: {med.medication} {med.dose}{med.unit} {med.route}")
        
        if patient.allergies and patient.allergies != ["No known drug allergies"]:
            summary_parts.append(f"Allergies: {', '.join(patient.allergies)}")
        
        critical_alerts = [alert for alert in recommendation.safety_alerts if alert['level'] == 'red']
        if critical_alerts:
            summary_parts.append(f"WARNING: {len(critical_alerts)} critical alerts identified")
        
        return ". ".join(summary_parts) + "."

    def _generate_summary_old(self, recommendation):
        """Generate clinical summary"""
        patient = recommendation.patient_info
        
        summary_parts = [
            f"Clinical Decision Support for {patient.name} (Age: {patient.age}, Weight: {patient.weight}kg)",
            f"Presenting symptoms: {', '.join(patient.symptoms) if patient.symptoms else 'Not specified'}",
            f"Severity assessment: {patient.severity}"
        ]
        
        if recommendation.cks_medication_dose:
            summary_parts.append(f"NICE CKS recommendation: {recommendation.cks_medication_dose.medication} {recommendation.cks_medication_dose.dose}{recommendation.cks_medication_dose.unit} {recommendation.cks_medication_dose.route}")
        
        if recommendation.local_medication_dose:
            summary_parts.append(f"Local guideline recommendation: {recommendation.local_medication_dose.medication} {recommendation.local_medication_dose.dose}{recommendation.local_medication_dose.unit} {recommendation.local_medication_dose.route}")
        
        critical_alerts = [alert for alert in recommendation.safety_alerts if alert['level'] == 'red']
        if critical_alerts:
            summary_parts.append(f"WARNING: {len(critical_alerts)} critical alerts identified")
        
        return "\n".join(summary_parts)