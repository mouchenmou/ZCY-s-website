 Implement `max_product`, which takes a list of numbers and returns the maximum product that can be formed by multiplying together non-consecutive elements of the list. Assume that all numbers in the input list are greater than or equal to 1.
 ```Python
    def max_product(s):
        """Return the maximum product of non-consecutive elements of s.
        
        >>> max_product([10, 3, 1, 9, 2]) # 10 * 9
        >>> 90
        >>> max_product([5, 10, 5, 10, 5]) # 5 * 5 * 5
        >>> 125
        >>> max_product([]) # The product of no numbers is 1
        >>> 1
        """
        我的解法：
        if len(s)==1: 
            return s[0]
        elif len(s)==0:
            return 1
        else:
            return max(s[0]*max_product(s[2:]), max_product(s[1:]))
 ```