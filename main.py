import matplotlib.animation as ani
import matplotlib.container
import matplotlib.pyplot as plt
import numpy as np


def bubble_sort(arr):
    for h in range(len(arr)):
        for i in range(1, len(arr) - h):
            # print(arr[i - 1], arr[i])
            if arr[i] < arr[i - 1]:
                arr[i - 1], arr[i] = arr[i], arr[i - 1]
            yield arr, (i,)

    return arr


def counting_sort(arr):
    if np.array(arr).dtype != np.int32:
        print("array needs to have only integers")
        return
    maxval = max(arr)
    counts = np.zeros(maxval + 1, dtype=np.int32)
    for i in arr:
        counts[i] += 1
    arr = np.zeros(len(arr), dtype=np.int32)
    curr = 0
    for i, count in enumerate(counts):
        arr[curr:curr + count] = i
        curr += count
    return arr


def recursive_mergesort(arr):
    l = len(arr)
    if l == 1:
        return arr
    if l == 2:
        if arr[0] > arr[1]:
            arr[0], arr[1] = arr[1], arr[0]
        return arr
    n = l // 2
    arr1 = recursive_mergesort(arr[:n])
    arr2 = recursive_mergesort(arr[n:])
    arr = np.zeros(len(arr))
    arr1idx = 0
    arr2idx = 0
    for maxidx in range(len(arr)):
        if arr1[arr1idx] < arr2[arr2idx]:
            arr[maxidx] = arr1[arr1idx]
            arr1idx += 1
        else:
            arr[maxidx] = arr2[arr2idx]
            arr2idx += 1
        if arr1idx == len(arr1):
            arr[maxidx + 1:] = arr2[arr2idx:]
            break
        elif arr2idx == len(arr2):
            arr[maxidx + 1:] = arr1[arr1idx:]
            break
    return arr

def selection_sort(arr):
    for i in range(len(arr) - 1, -1, -1):
        maxi = 0
        maxn = -12351235412354
        for j in range(i + 1):
            if arr[j] > maxn:
                maxn = arr[j]
                maxi = j
            yield arr, (j,)
        arr[i], arr[maxi] = arr[maxi], arr[i]
        yield arr, (maxi,)

def double_selection_sort(arr):
    mind = 0
    for mind in range(len(arr) // 2 ):
        maxi = 0
        mini = mind
        maxn = -12351235412354
        minn = 12341234123412
        for j in range(mind, len(arr) - mind):
            if arr[j] > maxn:
                maxn = arr[j]
                maxi = j
            elif arr[j] < minn:
                minn = arr[j]
                mini = j
            yield arr, (j,)
        end = len(arr) - 1 - mind
        if mini != end and mini != mind: #mini ni na robu
            if maxi != end and maxi != mind: #maxi ni na robu
                arr[end], arr[maxi] = arr[maxi], arr[end]
                arr[mind], arr[mini] = arr[mini], arr[mind]
            else: #maxi JE na robu
                arr[end], arr[maxi] = arr[maxi], arr[end]
                arr[mind], arr[mini] = arr[mini], arr[mind]
        else: #mini JE na robu
            if maxi != end and maxi != mind: #maxi NI na robu
                arr[mind], arr[mini] = arr[mini], arr[mind]
                arr[end], arr[maxi] = arr[maxi], arr[end]
            else: #oba sta na robu
                if mini > maxi:
                    arr[mini], arr[maxi] = arr[maxi], arr[mini]
        #print(arr)
        mind += 1
        yield arr, (maxi,mini)

def in_place_mergesort(arr):
    ...

def insertion_sort(arr):
    gap = 1
    for i in range(gap, len(arr)):
        temp = arr[i]
        j = i
        while j >= gap and arr[j - gap] > temp:
            arr[j] = arr[j - gap]
            j -= gap
            yield arr, (j, i)
        arr[j] = temp
        yield arr, (j, i)


def shell_sort(arr):
    steps = int(np.log2(len(arr))) + 1
    for g in range(steps):
        gap = len(arr) // (2 ** g)
        print(steps, g, gap)
        for i in range(gap, len(arr)):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
                yield arr, (j,i)
            arr[j] = temp
            yield arr, (j,i)

sort = bubble_sort
sort = selection_sort
sort = double_selection_sort
sort = shell_sort
sort = insertion_sort
# sort = counting_sort
# sort = recursive_mergesort
data = np.random.randint(50, size=(50))
# print(sort([2,12,32,2,3,5]))
#data = np.array(range(50))[::-1]
print(sort(data))

fig, ax = plt.subplots()
bar_container = ax.bar(np.arange(len(data)), data, lw=1,
                       ec="yellow", fc="green", alpha=0.5)
print(bar_container)
generator = sort(data)
def update(a):
    # simulate new data coming in
    arr, i = next(generator)
    for idx, (count, rect) in enumerate(zip(arr, bar_container.patches)):
        rect.set_height(count)
        if idx in i:
            rect.set_color('red')
        else:
            rect.set_color("green")
    return bar_container.patches


anim = ani.FuncAnimation(fig, update, interval=1/5000 ,repeat=False, blit=True)
plt.show()

