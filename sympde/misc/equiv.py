import torch

def assert_equiv(x, f, g, rtol = 1e-5, atol = 1e-8, print_only = False, print_precision = 6):
    """
    Test equivariance of a function f: X -> Y, where X and Y are vector spaces.
    """
    y1, y2 = f(g(x)), g(f(x))
    diff = torch.abs(y1 - y2)
    check = torch.allclose(y1, y2, rtol = rtol, atol = atol)
    message_fail = f"Equivariance test failed. \nMax difference:  {torch.max(diff):.{print_precision}} \nMean difference: {torch.mean(diff):.{print_precision}}"
    message_success = "Equivariance test passed."
    if print_only:
        print(message_success) if check is True else print(message_fail)
    else:
        assert check, message_fail