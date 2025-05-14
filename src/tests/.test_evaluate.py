import Evaluate as Evaluate

def test_evaluate_mse():
    assert Evaluate.meansquared_error([1, 2, 3], [1, 2, 3]) == 0

    # Test overlap conditions
    assert Evaluate.meansquared_error([1,2,3,4,5,6], [1,2,3]) == 0
    
