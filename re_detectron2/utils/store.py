import random
from collections import deque


class Store:
    def __init__(self, total_num_classes, items_per_class, shuffle=False, is_listable=False):
        self.shuffle = shuffle
        self.is_listable = is_listable
        self.items_per_class = items_per_class
        self.total_num_classes = total_num_classes
        self.store = [deque(maxlen=self.items_per_class) for _ in range(self.total_num_classes)]

    def add(self, items, class_ids):
        for index, class_id in enumerate(class_ids):
            self.store[class_id].append(items[index])

    def retrieve(self, class_id):
        if class_id != -1:
            items = []
            for item in self.store[class_id]:
                items.extend(list(item) if self.is_listable else item)
            if self.shuffle:
                random.shuffle(items)
            return items
        else:
            all_items = []
            for i in range(self.total_num_classes):
                items = []
                for item in self.store[i]:
                    items.append(list(item) if self.is_listable else item)
                all_items.append(items)
            return all_items

    def reset(self):
        self.store = [deque(maxlen=self.items_per_class) for _ in range(self.total_num_classes)]

    def __str__(self):
        string = ""

        for index, item in enumerate(self.store):
            string += f"\n Class {index} --> {len(list(item))} items"

        return f"{self.__class__.__name__}({string})"

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return sum([len(s) for s in self.store])


if __name__ == "__main__":
    store = Store(10, 3)
    store.add(('a', 'b', 'c', 'd', 'e', 'f'), (1, 1, 9, 1, 0, 1))
    store.add(('h',), (4,))

    print(store.retrieve(-1))
    print(store)
