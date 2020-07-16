#!/bin/env python3

class ParseTree: #
    def __init__(self,
                 depth=0,
                 text=None,
                 label=None,
                 children=None,
                 parent=None):
        self.label = label
        self.children = children if children != None else []
        self.text = text
        self.parent = parent
        self.depth = depth

    def isleaf(self):
        return len(self.children) == 0

    def leaves(self):
        """
        Returns all leaves of this tree.
        """
        if self.isleaf():
            return [self.text]
        else:
            return self.children[0].leaves() + self.children[1].leaves()

    def labeled_leaves(self):
        """
        Returns all leaves of this tree with its corresponding label.
        """
        if self.isleaf():
            return [(self.label, self.text)]
        else:
            return self.children[0].labeled_leaves() + self.children[1].labeled_leaves()

    def get_labeled_sentence(self):
        """
        Return all full sentences from this tree (one tree = one sentence i.e. parse the whole tree + the label)
        """
        if self.isleaf():
            return self.text
        elif self.parent != None:
            return self.children[0].get_labeled_sentence() + " " + self.children[1].get_labeled_sentence()

        return self.label, self.children[0].get_labeled_sentence() + " " + self.children[1].get_labeled_sentence()

    def copy(self):
        """
        Deep copy this tree
        """
        return ParseTree(
            depth = self.depth,
            text = self.text,
            label = self.label,
            children = self.children.copy() if self.children != None else [],
            parent = self.parent)

    def add_child(self, child):
        """
        Adds a child to the current tree.
        """
        self.children.append(child)
        child.parent = self

    def all_children(self):
        if len(self.children) > 0:
            left_children, right_children = self.children[0].all_children(), self.children[1].all_children()
            return [self] + left_children + right_children
        else:
            return [self]

    def lowercase(self):
        if len(self.children) > 0:
            for child in self.children:
                child.lowercase()
        else:
            self.text = self.text.lower()

    def to_lines(self):
        if len(self.children) > 0:
            left_lines, right_lines = self.children[0].to_lines(), self.children[1].to_lines()
            self_line = [left_lines[0] + " " + right_lines[0]]
            return self_line + left_lines + right_lines
        else:
            return [self.text]

    def to_labeled_lines(self):
        if len(self.children) > 0:
            left_lines, right_lines = self.children[0].to_labeled_lines(), self.children[1].to_labeled_lines()
            self_line = [(self.label, left_lines[0][1] + " " + right_lines[0][1])]
            return self_line + left_lines + right_lines
        else:
            return [(self.label, self.text)]

    def __str__(self):

        """
        String representation of a tree as visible in original corpus.

        print(tree)
        #=> '(2 (2 not) (3 good))'

        Outputs
        -------

            str: the String representation of the tree.

        """
        if len(self.children) > 0:
            rep = "(%d " % self.label
            for child in self.children:
                rep += str(child)
            return rep + ")"

        return ("(%d %s) " % (self.label, self.text))

class TreeParser:

    def attribute_text_label(self, node, current_word):
        """
        Tries to recover the label inside a string
        of the form '(3 hello)' where 3 is the label,
        and hello is the string. Label is not assigned
        if the string does not follow the expected
        format.

        Arguments:
        ----------
            node : ParseTree, current node that should possibly receive a label.
            current_word : str, input string.
        """
        node.text = current_word.lower()
        node.text = node.text.strip(" ")
        if len(node.text) > 0 and node.text[0].isdigit():
            split_sent = node.text.split(" ", 1)
            label = split_sent[0]
            if len(split_sent) > 1:
                text = split_sent[1]
                node.text = text

            if all(c.isdigit() for c in label):
                node.label = int(label)
            else:
                text = label + " " + text
                node.text = text

        if len(node.text) == 0:
            node.text = None


    def create_tree_from_string(self, line):
        """
        Parse and convert a string representation
        of an example into a ParseTree datastructure.

        Arguments:
        ----------
            line : str, string version of the tree.

        Returns:
        --------
            ParseTree : parsed tree.
        """
        depth = 0
        current_word = ""
        root = None
        current_node = root

        for char in line:
            if char == '(':
                if current_node is not None and len(current_word) > 0:
                    self.attribute_text_label(current_node, current_word)
                    current_word = ""
                depth += 1
                if depth > 1:
                    child = ParseTree(depth=depth)
                    current_node.add_child(child)
                    current_node = child
                else:
                    root = ParseTree(depth=depth)
                    current_node = root

            elif char == ')':
                if len(current_word) > 0:
                    self.attribute_text_label(current_node, current_word)
                    current_word = ""

                depth -= 1
                current_node = current_node.parent
            else:
                current_word += char
        if depth != 0:
            raise ParseError("Not an equal amount of closing and opening parentheses")

        return root

#tree = TreeParser()
#print(tree.create_tree_from_string("(2 (2 from) (2 innocence))").to_labeled_lines())
#print(tree.create_tree_from_string("(3 (2 It) (4 (4 (2 's) (4 (3 (2 a) (4 (3 lovely) (2 film))) (3 (2 with) (4 (3 (3 lovely) (2 performances)) (2 (2 by) (2 (2 (2 Buy) (2 and)) (2 Accorsi))))))) #(2 .)))").to_labeled_lines())


