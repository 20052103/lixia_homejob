"""
LRU Cache Design Solution

Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.
Both get() and put() operations must run in O(1) average time complexity.

The key insight is to use:
- A HashMap (dict) for O(1) key-value lookups
- A Doubly-Linked List to maintain the order of usage (LRU order)

When a key is accessed or updated, it moves to the front (most recently used).
When capacity is exceeded, we evict the tail (least recently used).
"""

from typing import Optional


class Node:
    """Node in a doubly-linked list"""
    def __init__(self, key: int = 0, value: int = 0):
        self.key = key
        self.value = value
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None


class LRUCache:
    """
    LRU Cache implementation using HashMap + Doubly-Linked List
    
    Time Complexity: O(1) for both get() and put()
    Space Complexity: O(capacity)
    """
    
    def __init__(self, capacity: int):
        """
        Initialize the LRU cache with positive size capacity.
        
        Args:
            capacity: Maximum number of items the cache can hold
        """
        self.capacity = capacity
        self.cache = {}  # key -> Node mapping for O(1) lookup
        
        # Dummy nodes for easier list manipulation
        self.head = Node()  # Most recently used
        self.tail = Node()  # Least recently used
        
        # Connect dummy nodes
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _add_to_head(self, node: Node) -> None:
        """Add a node right after head (most recently used position)"""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
    
    def _remove_node(self, node: Node) -> None:
        """Remove a node from the doubly-linked list"""
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _move_to_head(self, node: Node) -> None:
        """Move a node to head (mark as most recently used)"""
        self._remove_node(node)
        self._add_to_head(node)
    
    def get(self, key: int) -> int:
        """
        Return the value of the key if the key exists, otherwise return -1.
        
        Args:
            key: The key to look up
            
        Returns:
            The value associated with the key, or -1 if not found
        """
        if key not in self.cache:
            return -1
        
        # Node found, move it to head (mark as most recently used)
        node = self.cache[key]
        self._move_to_head(node)
        return node.value
    
    def put(self, key: int, value: int) -> None:
        """
        Update the value of the key if the key exists. Otherwise, add the 
        key-value pair to the cache. If the number of keys exceeds the 
        capacity, evict the least recently used key.
        
        Args:
            key: The key to insert/update
            value: The value to associate with the key
        """
        if key in self.cache:
            # Key exists, update its value and mark as most recently used
            node = self.cache[key]
            node.value = value
            self._move_to_head(node)
        else:
            # New key, create a new node
            new_node = Node(key, value)
            self.cache[key] = new_node
            self._add_to_head(new_node)
            
            # Check if capacity exceeded
            if len(self.cache) > self.capacity:
                # Remove the least recently used node (before tail)
                lru_node = self.tail.prev
                self._remove_node(lru_node)
                del self.cache[lru_node.key]


def test_lru_cache():
    """Test the LRU Cache implementation with the given example"""
    
    # Example from problem statement
    lru_cache = LRUCache(2)
    
    results = []
    
    # lru_cache.put(1, 1); // cache is {1=1}
    lru_cache.put(1, 1)
    results.append(None)
    
    # lru_cache.put(2, 2); // cache is {1=1, 2=2}
    lru_cache.put(2, 2)
    results.append(None)
    
    # lru_cache.get(1);    // return 1
    results.append(lru_cache.get(1))
    
    # lru_cache.put(3, 3); // LRU key was 2, evicts key 2, cache is {1=1, 3=3}
    lru_cache.put(3, 3)
    results.append(None)
    
    # lru_cache.get(2);    // returns -1 (not found)
    results.append(lru_cache.get(2))
    
    # lru_cache.put(4, 4); // LRU key was 1, evicts key 1, cache is {4=4, 3=3}
    lru_cache.put(4, 4)
    results.append(None)
    
    # lru_cache.get(1);    // return -1 (not found)
    results.append(lru_cache.get(1))
    
    # lru_cache.get(3);    // return 3
    results.append(lru_cache.get(3))
    
    # lru_cache.get(4);    // return 4
    results.append(lru_cache.get(4))
    
    expected = [None, None, 1, None, -1, None, -1, 3, 4]
    
    print("LRU Cache Test Results:")
    print(f"Results:  {results}")
    print(f"Expected: {expected}")
    print(f"Test Passed: {results == expected}")
    
    return results == expected


def test_lru_cache_edge_cases():
    """Test additional edge cases"""
    print("\n--- Edge Case Tests ---")
    
    # Test 1: Single capacity cache
    print("\nTest 1: Single capacity cache")
    lru = LRUCache(1)
    lru.put(1, 1)
    assert lru.get(1) == 1, "Should return 1"
    lru.put(2, 2)
    assert lru.get(1) == -1, "Should return -1 (evicted)"
    assert lru.get(2) == 2, "Should return 2"
    print("✓ Single capacity cache works correctly")
    
    # Test 2: Multiple accesses
    print("\nTest 2: Multiple accesses")
    lru = LRUCache(3)
    lru.put(1, 1)
    lru.put(2, 2)
    lru.put(3, 3)
    lru.get(1)  # Access 1, making 2 the LRU
    lru.put(4, 4)  # Should evict 2
    assert lru.get(2) == -1, "Should return -1 (evicted)"
    assert lru.get(1) == 1, "Should return 1"
    print("✓ Multiple accesses work correctly")
    
    # Test 3: Update existing key
    print("\nTest 3: Update existing key")
    lru = LRUCache(2)
    lru.put(1, 1)
    lru.put(2, 2)
    lru.put(1, 10)  # Update key 1, making 2 the LRU
    lru.put(3, 3)  # Should evict 2
    assert lru.get(1) == 10, "Should return 10 (updated)"
    assert lru.get(2) == -1, "Should return -1 (evicted)"
    assert lru.get(3) == 3, "Should return 3"
    print("✓ Update existing key works correctly")
    
    print("\nAll edge case tests passed! ✓")


if __name__ == "__main__":
    # Run the main test
    test_passed = test_lru_cache()
    print()
    
    # Run edge case tests
    test_lru_cache_edge_cases()
    
    print("\n" + "="*50)
    if test_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
