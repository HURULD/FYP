import RIRGen.Evaluate as Evaluate

def test_evaluate_mse():
    assert Evaluate.evaluate_mse([1, 2, 3], [1, 2, 3]) == 0

    # Test overlap conditions
    assert Evaluate.evaluate_mse([1,2,3,4,5,6], [1,2,3]) == 0
    
