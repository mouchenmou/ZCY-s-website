
### Case3
```python
Q: How do you write a doctest asserting that square(2) == 4?
Choose the number of the correct choice:
0) def square(x):
       '''
       square(2)
       4
       '''
       return x * x
1) def square(x):
       '''
       doctest: (2, 4)
       '''
       return x * x
2) def square(x):
       '''
       input: 2
       output: 4
       '''
       return x * x
3) def square(x):
       '''
       >>> square(2)
       4
       '''
       return x * x
?
```
![](附件/Pasted%20image%2020251217204455.png)

---
### Case4
```python
Choose the number of the correct choice:
0) For permanant debugging so you can have long term confidence in your code
1) To investigate the values of variables at certain points in your code
2) To ensure that certain conditions are true at certain points in your code
```
![](附件/Pasted%20image%2020251217204737.png)
![](附件/Pasted%20image%2020251217204826.png)

---
## Case5
```python
Q: How do you prevent the ok autograder from interpreting print statements as output?
Choose the number of the correct choice:
0) Print with # at the front of the outputted line
1) Print with 'DEBUG:' at the front of the outputted line
2) You don't need to do anything, ok only looks at returned values, not printed values
```
![](附件/Pasted%20image%2020251217205042.png)

---
## Case 6
```python
Q: What is the best way to open an interactive terminal to investigate a failing test for question sum_digits in assignment lab01?
Choose the number of the correct choice:
0) python3 -i lab01.py
1) python3 ok -q sum_digits -i
2) python3 ok -q sum_digits --trace
3) python3 ok -q sum_digits
```
![](附件/Pasted%20image%2020251217205224.png)

---
## Case7
```python
Q: Which of the following is NOT true?
Choose the number of the correct choice:
0) Code that returns a wrong answer instead of crashing is generally better as it at least works fine
1) Debugging is not a substitute for testing
2) It is generally bad practice to release code with debugging print statements left in
3) It is generally good practice to release code with assertion statements left in
4) Testing is very important to ensure robust code
```
![](附件/Pasted%20image%2020251217205705.png)
![](附件/Pasted%20image%2020251217205802.png)