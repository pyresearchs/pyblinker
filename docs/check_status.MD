- have cross check the landmark check and viz results shown in the  'landmark_check' and 'landmark_viz' folders.
- I have check the kinematic feature extra test/blink_features/kinematics/test_kinematics_ear_only_config.py, thing is working
- I have check the morphology test/blink_features/morphology/test_epoch_morphology_features_aggregation.py, and the flow and logic mostly follow the kinematic feature extraction, however, there are several redundet function
such as _style_windows,_infer_modality,_available_styles. next we should move this as common function to avoid redundancy.