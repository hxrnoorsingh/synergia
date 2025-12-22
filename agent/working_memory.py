class WorkingMemory:
    """
    A limited-capacity Working Memory (WM) store.
    Stores symbolic items (strings or tuples).
    """
    def __init__(self, capacity=4, alpha=0.0):
        self.capacity = capacity
        self.alpha = alpha # Forgetfulness/Noise parameter (unused in Step 1)
        self.items = [] # Simple list for now

    def add(self, item):
        """
        Adds an item to WM. If full, removes the oldest item (FIFO).
        """
        if item in self.items:
            # Refresh if already exists? Or duplicates allowed?
            # For this model, let's move to new (refresh)
            self.items.remove(item)
            self.items.append(item)
        else:
            if len(self.items) >= self.capacity:
                self.items.pop(0) # Remove oldest
            self.items.append(item)

    def contains(self, item):
        return item in self.items

    def get_all(self):
        return list(self.items)

    def clear(self):
        self.items = []

    def __repr__(self):
        return f"WM({self.items})"
