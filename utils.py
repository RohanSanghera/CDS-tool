import json
import re
from typing import Optional, Dict, Any

class JSONExtractor:
    """Extract and parse JSON from LLM responses"""
    
    @staticmethod
    def extract_json(response_text: str, debug: bool = False) -> Optional[str]:
        """ Extract JSON from LLM response """
        if debug:
            print(f" Response length {len(response_text)} chars")
            print(f"{response_text}")
        
        response_text = response_text.strip()
        
        if response_text.startswith('{') and response_text.endswith('}'):
            try:
                json.loads(response_text) 
                if debug:
                    print("Debug Found JSON object")
                return response_text
            except json.JSONDecodeError:
                if debug:
                    print("Debug - parsing failedg")
        
        json_str = JSONExtractor._extract_with_brace_matching(response_text, debug)
        if json_str:
            return json_str
        
        if debug:
            print("Debug; No valid JSON found")
        return None
    
    @staticmethod
    def _extract_with_brace_matching(text: str, debug: bool = False) -> Optional[str]:
        """Extract JSON using brace matching algorithm"""
        for start_idx in range(len(text)):
            if text[start_idx] == '{':
                brace_count = 0
                for end_idx in range(start_idx, len(text)):
                    if text[end_idx] == '{':
                        brace_count += 1
                    elif text[end_idx] == '}':
                        brace_count -= 1
                    
                    if brace_count == 0:
                        candidate = text[start_idx:end_idx + 1]
                        try:
                            json.loads(candidate) 

                            return candidate
                        except json.JSONDecodeError:
                            break  #
        return None
    
    @staticmethod
    def extract_and_parse(response_text: str, debug: bool = False) -> Optional[Dict[Any, Any]]:

        json_str = JSONExtractor.extract_json(response_text, debug)
        if json_str:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                if debug:
                    print(f"Debug: Failed to parse extract JSON: {e}")
        return None