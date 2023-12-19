import torch

def assert_equiv(x, f, g, postprocess = None, rtol = 1e-5, atol = 1e-8, print_only = False, print_precision = 6):
    """
    Test equivariance of a function f: X -> Y, where X and Y are vector spaces.
    """
    y1, y2 = f(g(x)), g(f(x))

    if postprocess is not None:
        y1, y2 = postprocess(y1), postprocess(y2)

    if y1.shape != y2.shape:
        print(f"Shape mismatch: {y1.shape} != {y2.shape}")
        return y1, y2

    diff = torch.abs(y1 - y2)
    check = torch.allclose(y1, y2, rtol = rtol, atol = atol)
    message_fail = f"Equivariance test failed. \nMax difference:  {torch.max(diff):.{print_precision}} \nMean difference: {torch.mean(diff):.{print_precision}}"
    message_success = "Equivariance test passed."
    if print_only:
        print(message_success) if check is True else print(message_fail)
    else:
        assert check, message_fail
    return y1, y1

def allclose_flat(a, b):
    return torch.allclose(a.flatten(), b.flatten(), atol=1e-6, rtol=1e-6), (a - b)

def check(a, b, string):
    print(string, a.shape)
    print(len(string) * ' ', b.shape)
    # all_close_flat(a, b)