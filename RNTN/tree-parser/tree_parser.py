class ParseTree:
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
        if self.isleaf():
            return [self.text]
        else:
            return self.children[0].leaves() + self.children[1].leaves()

    def labeled_leaves(self):
        if self.isleaf():
            return [(self.label, self.text)]
        else:
            return self.children[0].labeled_leaves() + self.children[1].labeled_leaves()

    def get_labeled_sentences(self):
        if self.isleaf():
            return self.text
        elif self.parent != None:
            return self.children[0].get_labeled_sentences() + " " + self.children[1].get_labeled_sentences()

        return self.label, self.children[0].get_labeled_sentences() + " " + self.children[1].get_labeled_sentences()

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def all_children(self):
        if len(self.children) > 0:
            left_children, right_children = self.children[0].all_children(), self.children[1].all_children()
            return [self] + left_children + right_children
        else:
            return [self]

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
        if len(self.children) > 0:
            rep = "(%d " % self.label
            for child in self.children:
                rep += str(child)
            return rep + ")"

        return ("(%d %s) " % (self.label, self.text))

class TreeParser:

    def preprocess_string(self, text):
        return text.lower()\
               .replace("-lrb-", "(")\
               .replace("-rrb-", ")")\
               .replace("-lcb-", "{")\
               .replace("-rcb-", "}")\
               .replace("-lsb-", "[")\
               .replace("-rsb-", "]")\
               .strip(" ")

    def attribute_text_label(self, node, current_word, binary=False):
        node.text = self.preprocess_string(current_word)
        if len(node.text) > 0 and node.text[0].isdigit():
            split_sent = node.text.split(" ", 1)
            label = split_sent[0]
            if len(split_sent) > 1:
                text = split_sent[1]
                node.text = text

            if all(c.isdigit() for c in label):
                if binary:
                    node.label = TreeParser.map_label_to_binary(int(label))
                else:
                    node.label = int(label)
            else:
                text = label + " " + text
                node.text = text

        if len(node.text) == 0:
            node.text = None

    @staticmethod
    def map_label_to_binary(label):
        if label <= 2:
            return 0
        return 1

    def create_tree_from_string(self, line, binary=False):
        depth = 0
        current_word = ""
        root = None
        current_node = root

        for char in line:
            if char == '(':
                if current_node is not None and len(current_word) > 0:
                    self.attribute_text_label(current_node, current_word, binary=binary)
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
                    self.attribute_text_label(current_node, current_word, binary=binary)
                    current_word = ""

                depth -= 1
                current_node = current_node.parent
            else:
                current_word += char
        if depth != 0:
            raise ParseError("Not an equal number of closing and opening brackets")

        return root



