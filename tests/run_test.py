from test_pgd import test_pgd_runs, test_pgd_within_epsilon
from test_sdp import test_sdp_runs, test_sdp_deterministic, test_sdp_soundness_property
from test_shapes import test_model_forward, test_sdp_input_shapes

def run_tests():
    tests = [
        test_sdp_runs,
        test_pgd_within_epsilon,
        test_pgd_runs,
        test_sdp_deterministic,
        test_sdp_soundness_property,
        test_model_forward,
        test_sdp_input_shapes
    ]

    for t in tests:
        try:
            t()
            print(f"{t.__name__}: OK")
        except Exception as e:
            print(f"{t.__name__}: FAIL -> {e}")


if __name__ == "__main__":
    run_tests()