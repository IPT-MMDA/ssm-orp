import numpy as np
import scipy.stats as st
import math

def calculate_stats(acc_list):
    mean = np.mean(acc_list)
    sem = st.sem(acc_list) 
    ci_lower, ci_upper = st.t.interval(confidence=0.95, df=len(acc_list)-1, loc=mean, scale=sem)
    margin = ci_upper - mean
    return mean, ci_lower, ci_upper, margin

def test_confidence_interval_logic():
    dummy_accuracies = [70.0, 75.0, 80.0] 
    mean, ci_lower, ci_upper, margin = calculate_stats(dummy_accuracies)
    
    assert math.isclose(mean, 75.0), f"Expected mean 75.0, got {mean}"
    assert ci_lower < mean < ci_upper, "Confidence interval bounds are incorrect!"
    assert margin > 0, "Margin of error must be positive!"