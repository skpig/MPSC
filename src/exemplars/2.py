from typing import List

def minNumber(nums1: List[int], nums2: List[int]) -> int:
    """
    Given two arrays of unique digits nums1 and nums2, return the smallest number that contains at least one digit from each array.

    For example:

    minNumber([4, 1, 3], [5, 7])  ==> 15
    Explanation: The number 15 contains the digit 1 from nums1 and the digit 5 from nums2. It can be proven that 15 is the smallest number we can have.

    minNumber([3, 5, 2, 6], [3, 1, 7]) ==> 3
    Explanation: The number 3 contains the digit 3 which exists in both arrays.
    """
    pass

# Pre-conditions
def preconditions(nums1, nums2):
    assert isinstance(nums1, list), "nums1 must be a list"
    assert isinstance(nums2, list), "nums2 must be a list"
    digits = set(range(10))
    assert all(item in digits for item in nums1), "nums1 must be a list of digits"
    assert all(item in digits for item in nums2), "nums2 must be a list of digits"

# Post-conditions
def postconditions(nums1, nums2, output):
    assert isinstance(output, int), "output must be an integer"
    assert output >= 0, "output must be a non-negative integer"
    assert any(str(digit) in str(output) for digit in nums1), "output must contain at least one digit from nums1"
    assert any(str(digit) in str(output) for digit in nums2), "output must contain at least one digit from nums2"

    for digit1 in nums1:
        for digit2 in nums2:
            assert output <= int(str(digit1) + str(digit2)) and output <= int(str(digit2) + str(digit1)), "output must be the smallest number that contains at least one digit from nums1 and nums2"

# Test inputs
test_inputs = [([0, 1, 2], [3, 4, 5]),
               ([0], [9]),
               ([0, 1, 2], [0, 3, 4]),
               ([1, 6, 2, 9, 8], [4, 5, 1, 2])]