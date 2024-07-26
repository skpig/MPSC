from typing import List

def maximumCostSubstring(s: str, chars: str, vals: List[int]) -> str:
    """
    You are given a string s, a string chars of distinct characters and an integer array vals of the same length as chars.

    The cost of the substring is the sum of the values of each character in the substring. The cost of an empty string is considered 0.

    The value of the character is defined in the following way:
    1. If the character is not in the string chars, then its value is its corresponding position (1-indexed) in the alphabet. For example, the value of 'a' is 1, the value of 'b' is 2, and so on. The value of 'z' is 26.
    2. Otherwise, assuming i is the index where the character occurs in the string chars, then its value is vals[i].
    
    Return the substring of the string s with the maximum cost among all substrings.

    Example 1:
        Input: s = "adaa", chars = "d", vals = [-1000]
        Output: "aa"
        Explanation: The value of the characters "a" and "d" is 1 and -1000 respectively. The substring with the maximum cost is "aa" and its cost is 1 + 1 = 2.

    Example 2:
        Input: s = "abc", chars = "abc", vals = [-1,-1,-1]
        Output: ""
        Explanation: The value of the characters "a", "b" and "c" is -1, -1, and -1 respectively. The substring with the maximum cost is the empty substring "" and its cost is 0.
    """
    pass

# Pre-conditions
def preconditions(s, chars, vals):
    assert isinstance(s, str) and isinstance(chars, str) and isinstance(vals, list)
    assert all(isinstance(val, int) for val in vals), "vals must be a list of integers"
    assert len(chars) == len(vals)

# Post-conditions
def postconditions(s, chars, vals, output):
    assert isinstance(output, str), "output must be a string"
    assert s.find(output) != -1, "output must be a substring of s"
    
    # calculate cost of output
    def get_val(substring):
        cost = 0
        for char in substring:
            if char in chars:
                cost += vals[chars.index(char)]
            else:
                cost += ord(char) - ord('a') + 1
        return cost
    
    assert get_val(output) >= 0, "output must have a cost greater than or equal to empty string"
    assert get_val(output) >= max(get_val(s[i:j]) for i in range(len(s)) for j in range(i + 1, len(s) + 1)), "output must be the substring of s with the maximum cost"

# Test inputs
test_inputs = [("aeez", "kzp", [-1000, 20, 10]),
               ("zzzattttboooo", "zto", [1, 2, 3]),
               ("abababababab", "b", [-2])]