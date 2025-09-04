try:
    from src.core.body_reconstruction import BodyReconstructor
    from src.core.garment_fitting import GarmentFitter
    from src.core.rendering import PhotorealisticRenderer
    from src.core.ai_enhancement import AIEnhancer
    print('✅ All 3D modules imported successfully')
    
    body_reconstructor = BodyReconstructor()
    garment_fitter = GarmentFitter()
    renderer = PhotorealisticRenderer()
    ai_enhancer = AIEnhancer()
    print('✅ All 3D modules initialized successfully')
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
