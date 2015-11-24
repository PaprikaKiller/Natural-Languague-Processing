from operator import lt


class Node(object):
    left = None
    right = None
    item = None
    weight = 0

    def __init__(self, i, w):
        self.item = i
        self.weight = w

    def set_children(self, ln, rn):
        self.left = ln
        self.right = rn

    def __repr__(self):
        return "%s - %s — %s _ %s" % (self.item, self.weight, self.left, self.right)

    def __lt__(self, other):
        return lt(self.weight, other.weight)
