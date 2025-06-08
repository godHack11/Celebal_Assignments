# Implement a Linked List in Python Using OOP and Delete the Nth Node

# Create a Python program that implements a singly linked list using Object-Oriented Programming (OOP) principles. 
# Your implementation should include the following: A Node class to represent each node in the list. A LinkedList class to manage the 
# nodes, with methods to: Add a node to the end of the list Print the list Delete the nth node (where n is a 1-based index) Include 
# exception handling to manage edge cases such as: Deleting a node from an empty list Deleting a node with an index out of range Test 
# your implementation with at least one sample list.



class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def add_node(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
        else:
            temp = self.head
            while temp.next:
                temp = temp.next
            temp.next = new_node

    def print_list(self):
        if self.head is None:
            print("List is empty.")
        else:
            temp = self.head
            while temp:
                print(temp.data, end=" -> " if temp.next else "\n")
                temp = temp.next

    def delete_nth_node(self, n):
        if self.head is None:
            print("Error: Cannot delete from an empty list.")
            return

        if n <= 0:
            print("Error: Index should be 1 or greater.")
            return

        if n == 1:
            self.head = self.head.next
            return

        temp = self.head
        count = 1
        while temp is not None and count < n - 1:
            temp = temp.next
            count += 1

        if temp is None or temp.next is None:
            print("Error: Index out of range.")
            return

        temp.next = temp.next.next

my_list = LinkedList()
my_list.add_node(10)
my_list.add_node(20)
my_list.add_node(30)
my_list.add_node(40)

print("Original List:")
my_list.print_list()

my_list.delete_nth_node(2)

print("After deleting 2nd node:")
my_list.print_list()

# What This Code Does:

# It creates and manages a singly linked list using Python and OOP (Object-Oriented Programming). It allows you to:
# 1. Add items (nodes) at the end
# 2. Print all items in the list
# 3. Delete any item by its position (like 1st, 2nd, 3rd…)
# 4. Handles errors properly (like empty list or wrong position)

# Main Parts of the Code:
# 1. Node class

# This is like a building block. Each node stores:
# • data → the value (like 10, 20)
# • next → a pointer to the next node (or None if it’s the last one)

# 2. LinkedList class

# This manages the whole list. It has:
# • A starting point called head
# • A method add_node(data) to add a value to the end
# • A method print_list() to display the list
# • A method delete_nth_node(n) to delete the node at position n







    
