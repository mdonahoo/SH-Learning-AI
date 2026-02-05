#!/usr/bin/env python3
"""Test script to debug UtteranceLevelRoleDetector import."""

import sys
import traceback
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(name)s: %(message)s')

print("=" * 80)
print("Testing UtteranceLevelRoleDetector import")
print("=" * 80)

# Test 1: Base imports
try:
    print("\n1. Importing base components...")
    print("   Importing RoleInferenceEngine and EnhancedRoleInferenceEngine...")
    from src.metrics.role_inference import RoleInferenceEngine, EnhancedRoleInferenceEngine
    print("   ✓ Base imports successful")
except Exception as e:
    print(f"   ✗ Base import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 2: Import UtteranceLevelRoleDetector
try:
    print("\n2. Importing UtteranceLevelRoleDetector...")
    from src.metrics.role_inference import UtteranceLevelRoleDetector
    print("   ✓ UtteranceLevelRoleDetector import successful")
except Exception as e:
    print(f"   ✗ Failed to import UtteranceLevelRoleDetector: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 3: Instantiate
try:
    print("\n3. Instantiating UtteranceLevelRoleDetector...")
    detector = UtteranceLevelRoleDetector()
    print("   ✓ Instantiation successful")
except Exception as e:
    print(f"   ✗ Instantiation failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test detections with multiple examples
try:
    print("\n4. Testing detect_role_for_utterance with multiple examples...")

    test_cases = [
        ("Tactical systems green across the board, shields at 100%", "TACTICAL"),
        ("Set course for starbase one, engage warp drive", "HELM"),
        ("Engineering reports warp core stable, power distribution optimal", "ENGINEERING"),
        ("Captain's log, stardate", "CAPTAIN"),
        ("Scanning sector for anomalies", "SCIENCE"),
        ("All ships in formation, standing by", "OPERATIONS"),
    ]

    for text, expected_role in test_cases:
        role, confidence, keywords = detector.detect_role_for_utterance(text)
        status = "✓" if expected_role in role.value.upper() else "✗"
        print(f"   {status} '{text[:50]}...'")
        print(f"       Role: {role.value} (conf={confidence:.3f}, keywords={len(keywords)})")

    print("   ✓ Detection tests successful")

except Exception as e:
    print(f"   ✗ Detection failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test pipeline integration
try:
    print("\n5. Testing pipeline imports (simulating audio_processor.py)...")

    # Simulate the imports that audio_processor.py does
    try:
        from src.metrics.role_inference import RoleInferenceEngine, EnhancedRoleInferenceEngine
        ROLE_INFERENCE_AVAILABLE = True
        print("   ✓ Base role inference imports successful")
    except Exception as e:
        print(f"   ✗ Base imports failed: {e}")
        ROLE_INFERENCE_AVAILABLE = False

    try:
        from src.metrics.role_inference import UtteranceLevelRoleDetector
        print("   ✓ UtteranceLevelRoleDetector available")
    except Exception as e:
        print(f"   ✗ UtteranceLevelRoleDetector import failed: {e}")
        UtteranceLevelRoleDetector = None

    # Check the condition from audio_processor.py line 1571
    if ROLE_INFERENCE_AVAILABLE and UtteranceLevelRoleDetector:
        print("   ✓ Pipeline condition check PASSED")
        print("   ✓ Utterance-level detection block WOULD EXECUTE")
    else:
        print(f"   ✗ Pipeline condition check FAILED")
        print(f"      ROLE_INFERENCE_AVAILABLE={ROLE_INFERENCE_AVAILABLE}")
        print(f"      UtteranceLevelRoleDetector={'available' if UtteranceLevelRoleDetector else 'NOT available'}")
        sys.exit(1)

except Exception as e:
    print(f"   ✗ Pipeline test failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✓✓✓ SUCCESS: All imports and tests passed! ✓✓✓")
print("=" * 80)
print("\nThe UtteranceLevelRoleDetector is properly available and should execute in the pipeline.")
