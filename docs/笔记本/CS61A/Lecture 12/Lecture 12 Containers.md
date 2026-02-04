## 1. Box-and-Pointer Notation in Environment Diagrams
- List are represented as a row of index-labeled adjacent boxes, one per element.
- Each box either contains a primitive value or points to a compound value.
![](附件/Pasted%20image%2020260203211702.png)

---

## 2. Slicing
![](附件/Pasted%20image%2020260203212028.png)

---
## 3. Processing Container Values

![](附件/Pasted%20image%2020260203213518.png)

---

## 4. String
Lists, you can only look for one element at a time, but in strings, you can look for consecutive letters.
![](附件/Pasted%20image%2020260204135856.png)

---

## 5. Dictionary

- Dictionaries are a built-in data type in Python that hold pairs of a key, which is what you use to look something up, which is what you use to look up a value, and the corresponding value.
- For a dictionary, the only values that we can put in the square brackets in order to look them up are keys.
![](附件/Pasted%20image%2020260204141154.png)
![](附件/Pasted%20image%2020260204141417.png)

### Limitations on Dictionary

Dictionaries are collections of key-value pairs

Dictionary keys do have two restrictions :

- A key of a dictionary cannot be a list or a dictionary (or any mutable type)
- Two keys cannot be equal; There can be at most one value for a given key

!!! example
    ![](附件/Pasted%20image%2020260204143827.png)