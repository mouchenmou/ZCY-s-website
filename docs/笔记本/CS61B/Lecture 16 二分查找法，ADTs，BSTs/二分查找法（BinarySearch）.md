```c
int binarySearch ( int a[], int n, int x){
    int left = 0;
    int right = n-1;
    int mid = (right+left)/2;
    while(right>=left){
        if(a[mid]==x){
            return mid;
        }else if(a[mid]<x){
            left = mid+1;
        }else{
            right = mid-1;
        }
    }
    return 1;
}
```
## 最坏的情况下，它的时间复杂度是Θ(logN)
