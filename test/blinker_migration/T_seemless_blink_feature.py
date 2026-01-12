"""
We will modify the blink properties coding so that it can work seamlessly with code under
pyblinker/blink_features/morphology and pyblinker/blink_features/kinematics
This is to reduce the code redundancy and improve maintainability.
"""
from pyblinker.blink_features.waveform_features.extract_blink_properties import BlinkProperties
from pyblinker.utils.blinker_feature import save_blinkprops_pickle,load_blinkprops_pickle,replay_and_assert_blinkprops

fx = load_blinkprops_pickle("blinkprops_case01.pkl")
replay_and_assert_blinkprops(fx, BlinkProperties)