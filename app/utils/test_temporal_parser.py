
import sys
import os
from datetime import datetime

# Adjust path to import app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app.utils.temporal_parser import parse_temporal_intent
from app.config.time import HOSPITAL_TZ

def run_tests():
    now = datetime(2025, 12, 11, 10, 0, 0, tzinfo=HOSPITAL_TZ)
    print(f"Testing with NOW = {now}\n")

    test_cases = [
        # Positive cases
        ("tras la cirugía del 17/11/2023", "RANGE", "2023-11-17", True),
        ("despues del 2024-01-01", "RANGE", "2024-01-01", True),
        ("a partir de 15/05/2024", "RANGE", "2024-05-15", True),
        ("antes del 01/01/2024", "RANGE", None, "2024-01-01"),
        
        # Robustness cases (Negatives) - Should be NONE or not match the "tras" part
        ("tras la complicación pulmonar", "NONE", None, None), 
        ("que pasó despues de la cirugía", "NONE", None, None),
        ("síntomas previos a la internación", "NONE", None, None),
        
        # Mixed/Existing cases
        ("hoy", "DAY", "2025-12-11", "2025-12-12"),
    ]

    for q, expected_type, exp_start, exp_end in test_cases:
        res = parse_temporal_intent(q, now=now)
        
        if expected_type == "NONE":
            success = (res is None)
        else:
            success = (res is not None and res.temporal_intent_type == expected_type)
            if success and exp_start:
                # Check start date mostly matches (string check usually ensures it)
                 if exp_start not in str(res.start_iso): success = False
            if success and exp_end is True: # Just check it exists/is future
                 if not res.end_iso: success = False
            elif success and isinstance(exp_end, str):
                 if exp_end not in str(res.end_iso): success = False

        status = "PASS" if success else "FAIL"
        print(f"[{status}] Query: '{q}'")
        if not success:
            print(f"   Expected: {expected_type}, Start:{exp_start}, End:{exp_end}")
            print(f"   Got: {res}")

if __name__ == "__main__":
    run_tests()
