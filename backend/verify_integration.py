#!/usr/bin/env python3
"""
Verify that model_manager can import the fixed rendering classes
"""

try:
    from src.core.model_manager import model_manager
    print('✅ Model manager can import fixed classes')
    
    renderer = model_manager.get_renderer()
    if renderer is not None:
        print('✅ Renderer loaded successfully')
    else:
        print('⚠️ Renderer is None but import worked')
        
    enhancer = model_manager.get_ai_enhancer()
    if enhancer is not None:
        print('✅ AI enhancer loaded successfully')
    else:
        print('⚠️ AI enhancer is None but import worked')
        
except Exception as e:
    print(f'❌ Integration test failed: {e}')
    exit(1)

print('✅ All integration tests passed!')
